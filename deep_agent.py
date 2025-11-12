import asyncio
from typing import Literal
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    get_buffer_string,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
)
from state import (
    AgentState,
    SupervisorState,
    ResearcherState,
    UserQueryState,
    ResearchBriefState,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
)
from configurations import Configuration
from utils import (
    tavily_search,
    think_tool,
    get_api_key_for_model,
    get_today_date,
    get_notes_from_tool_calls,
    is_token_limit_exceeded,
    get_model_token_limit,
    get_all_tools,
    openai_websearch_called,
    anthropic_websearch_called,
    remove_up_to_last_ai_message,
)
from prompts import (
    clarify_user_prompt,
    research_question_prompt,
    research_system_prompt,
    compress_research_system_prompt,
    compress_research_simple_human_message,
    research_supervisor_prompt,
    final_report_generation_prompt,
)

configurable_model = init_chat_model(configurable_fields=("model", "max_tokens", "api_key"))


async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__", "write_research_brief"]]:
    configurables = Configuration.from_runnable_config(config)
    if not configurables.allow_clarification:
        return Command(goto="write_research_brief")
    model_config = {
        "model": configurables.research_model,
        "max_tokens": configurables.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurables.research_model, config),
        "tags": ["langsmith:nostream"],
    }
    user_clarification_model = (
        configurable_model.with_config(model_config)
        .with_structured_output(UserQueryState)
        .with_retry(configurables.max_failed_retry)
    )
    prompt = clarify_user_prompt.format(messages=get_buffer_string(state["messages"]), date=get_today_date())
    response = await user_clarification_model.ainvoke([HumanMessage(content=prompt)])
    try:
        if response.need_clarification:
            return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
        else:
            return Command(goto="write_research_brief", update={"messages": [AIMessage(content=response.verification)]})
    except Exception:
        return Command(goto=END)


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    prompt = research_question_prompt.format(messages=get_buffer_string(state["messages"]), date=get_today_date())
    configurables = Configuration.from_runnable_config(config)
    model_config = {
        "model": configurables.research_model,
        "max_tokens": configurables.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurables.research_model, config),
        "tags": ["langsmith:nostream"],
    }
    research_briefing_model = (
        configurable_model.with_config(model_config)
        .with_structured_output(ResearchBriefState)
        .with_retry(configurables.max_failed_retry)
    )
    response = await research_briefing_model.ainvoke([HumanMessage(content=prompt)])
    supervisor_prompt = research_supervisor_prompt.format(
        date=get_today_date(),
        max_concurrent_research_units=configurables.max_concurrent_research_units,
        max_researcher_iterations=configurables.max_researcher_iterations,
    )
    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_prompt),
                    HumanMessage(content=response.research_brief),
                ],
            },
        },
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    configurables = Configuration.from_runnable_config(config)
    model_config = {
        "model": configurables.research_model,
        "max_tokens": configurables.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurables.research_model, config),
        "tags": ["langsmith:nostream"],
    }
    supervisor_tools = [ConductResearch, ResearchComplete, think_tool]
    supervisor_model = (
        configurable_model.with_config(model_config)
        .bind_tools(supervisor_tools)
        .with_retry(configurables.max_failed_retry)
    )
    response = await supervisor_model.ainvoke(state.get("supervisor_messages"))
    return Command(
        goto="supervisor_tools",
        update={"supervisor_messages": [response], "research_iterations": state.get("research_iterations", 0) + 1},
    )


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls
    )
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={"notes": get_notes_from_tool_calls(supervisor_messages), "research_brief": state.get("research_brief", "")},
        )
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    think_tool_calls = [tool_call for tool_call in most_recent_message.tool_calls if tool_call["name"] == "think_tool"]
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(
            ToolMessage(content=f"Reflection recorded: {reflection_content}", name="think_tool", tool_call_id=tool_call["id"])
        )
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls if tool_call["name"] == "ConductResearch"
    ]
    if conduct_research_calls:
        try:
            allowed_conduct_research_calls = conduct_research_calls[: configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units :]
            research_tasks = [
                researcher_subgraph.ainvoke(
                    {
                        "researcher_messages": [HumanMessage(content=tool_call["args"]["research_topic"])],
                        "research_topic": tool_call["args"]["research_topic"],
                    },
                    config,
                )
                for tool_call in allowed_conduct_research_calls
            ]
            tool_results = await asyncio.gather(*research_tasks)
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(
                    ToolMessage(
                        content=observation.get(
                            "compressed_research", "Error synthesizing research report: Maximum retries exceeded"
                        ),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(
                    ToolMessage(
                        content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                        name="ConductResearch",
                        tool_call_id=overflow_call["id"],
                    )
                )
            raw_notes_concat = "\n".join(["\n".join(observation.get("raw_notes", [])) for observation in tool_results])
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
        except Exception as e:
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                return Command(
                    goto=END,
                    update={"notes": get_notes_from_tool_calls(supervisor_messages), "research_brief": state.get("research_brief", "")},
                )
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(goto="supervisor", update=update_payload)


supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()


async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError("No tools found to conduct research.")
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }
    researcher_prompt = research_system_prompt.format(mcp_prompt=configurable.mcp_prompt or "", date=get_today_date())
    research_model = (
        configurable_model.bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    return Command(
        goto="researcher_tools",
        update={"researcher_messages": [response], "tool_call_iterations": state.get("tool_call_iterations", 0) + 1},
    )


async def execute_tool_safely(tool, args, config):
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = openai_websearch_called(most_recent_message) or anthropic_websearch_called(most_recent_message)
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool for tool in tools
    }
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    tool_outputs = [
        ToolMessage(content=observation, name=tool_call["name"], tool_call_id=tool_call["id"])
        for observation, tool_call in zip(observations, tool_calls)
    ]
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls)
    if exceeded_iterations or research_complete_called:
        return Command(goto="compress_research", update={"researcher_messages": tool_outputs})
    return Command(goto="researcher", update={"researcher_messages": tool_outputs})


async def compress_research(state: ResearcherState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config(
        {
            "model": configurable.compression_model,
            "max_tokens": configurable.compression_model_max_tokens,
            "api_key": get_api_key_for_model(configurable.compression_model, config),
            "tags": ["langsmith:nostream"],
        }
    )
    researcher_messages = state.get("researcher_messages", [])
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    synthesis_attempts = 0
    max_attempts = 3
    while synthesis_attempts < max_attempts:
        try:
            compression_prompt = compress_research_system_prompt.format(date=get_today_date())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            response = await synthesizer_model.ainvoke(messages)
            raw_notes_content = "\n".join(
                [str(message.content) for message in filter_messages(researcher_messages, include_types=["tool", "ai"])]
            )
            return {"compressed_research": str(response.content), "raw_notes": [raw_notes_content]}
        except Exception as e:
            synthesis_attempts += 1
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            continue
    raw_notes_content = "\n".join(
        [str(message.content) for message in filter_messages(researcher_messages, include_types=["tool", "ai"])]
    )
    return {"compressed_research": "Error synthesizing research report: Maximum retries exceeded", "raw_notes": [raw_notes_content]}


researcher_builder = StateGraph(ResearcherState, output=ResearcherOutputState, config_schema=Configuration)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()


async def final_report_generation(state: AgentState, config: RunnableConfig):
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"],
    }
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    while current_retry <= max_retries:
        try:
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_date(),
            )
            final_report = await configurable_model.with_config(writer_model_config).ainvoke(
                [HumanMessage(content=final_report_prompt)]
            )
            return {"final_report": final_report.content, "messages": [final_report], **cleared_state}
        except Exception as e:
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                if current_retry == 1:
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, model context length unknown. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state,
                        }
                    findings_token_limit = model_token_limit * 4
                else:
                    findings_token_limit = int(findings_token_limit * 0.9)
                findings = findings[:findings_token_limit]
                continue
            else:
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state,
                }
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state,
    }


deep_researcher_builder = StateGraph(AgentState, input=AgentState, config_schema=Configuration)
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)
deep_researcher = deep_researcher_builder.compile()

