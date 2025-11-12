# src/open_deep_research/deep_agent.py

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
    ResearchState,            
    UserQueryState,
    ResearchBriefState,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,    
)
from configurations import Configuration
from agent_utils import (
    tavily_search,
    think_tool,
    get_api_key_for_model,
    get_today_date,
    get_notes_from_tool_calls,
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


async def clarify_with_user(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["__end__", "write_research_brief"]]:
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
        .with_retry()
    )

    prompt = clarify_user_prompt.format(
        messages=get_buffer_string(state["messages"]),
        date=get_today_date(),
    )
    response = await user_clarification_model.ainvoke([HumanMessage(content=prompt)])

    try:
        if response.need_clarification:
            return Command(
                goto=END, update={"messages": [AIMessage(content=response.question)]}
            )
        else:
            return Command(
                goto="write_research_brief",
                update={"messages": [AIMessage(content=response.verification)]},
            )
    except Exception:
        return Command(goto=END)


async def write_research_brief(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["research_supervisor"]]:
    prompt = research_question_prompt.format(
        messages=get_buffer_string(state["messages"]),
        date=get_today_date(),
    )
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
        .with_retry()
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
        .with_retry()
    )

    response = await supervisor_model.ainvoke(state.get("supervisor_messages"))

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
        },
    )


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    configurables = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    exceeded_allowed_iterations = research_iterations > configurables.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls
    )

    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
            },
        )

    all_tool_messages = []
    update_payload = {"supervisor_messages": []}

    think_tool_calls = [tc for tc in most_recent_message.tool_calls if tc["name"] == "think_tool"]
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(
            ToolMessage(
                content=f"Reflection recorded: {reflection_content}",
                name="think_tool",
                tool_call_id=tool_call["id"],
            )
        )

    conduct_research_calls = [tc for tc in most_recent_message.tool_calls if tc["name"] == "ConductResearch"]
    if conduct_research_calls:
        try:
            allowed = conduct_research_calls[: configurables.max_concurrent_research_units]
            overflow = conduct_research_calls[configurables.max_concurrent_research_units :]

            research_tasks = [
                researcher_subgraph.ainvoke(
                    {
                        "researcher_messages": [HumanMessage(content=tool_call["args"]["research_topic"])],
                        "research_topic": tool_call["args"]["research_topic"],
                    },
                    config,
                )
                for tool_call in allowed
            ]
            tool_results = await asyncio.gather(*research_tasks)

            for observation, tool_call in zip(tool_results, allowed):
                all_tool_messages.append(
                    ToolMessage(
                        content=observation.get(
                            "compressed_research", "Error synthesizing research report: Maximum retries exceeded"
                        ),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )

            for overflow_call in overflow:
                all_tool_messages.append(
                    ToolMessage(
                        content=(
                            f"Error: Did not run this research as you have already exceeded the maximum number "
                            f"of concurrent research units. Please try again with "
                            f"{configurables.max_concurrent_research_units} or fewer research units."
                        ),
                        name="ConductResearch",
                        tool_call_id=overflow_call["id"],
                    )
                )

            raw_notes_concat = "\n".join(["\n".join(obs.get("raw_notes", [])) for obs in tool_results])
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]

        except Exception:
            return Command(
                goto=END,
                update={
                    "notes": get_notes_from_tool_calls(supervisor_messages),
                    "research_brief": state.get("research_brief", ""),
                },
            )

    update_payload["supervisor_messages"] = all_tool_messages
    return Command(goto="supervisor", update=update_payload)

supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()



async def researcher(state: ResearchState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    configurables = Configuration.from_runnable_config(config)

    # Bind only the tools you actually have
    tools = [tavily_search, think_tool, ResearchComplete]

    research_model_config = {
        "model": configurables.research_model,
        "max_tokens": configurables.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurables.research_model, config),
        "tags": ["langsmith:nostream"],
    }

    researcher_prompt = research_system_prompt.format(mcp_prompt=configurables.mcp_prompt or "", date=get_today_date())

    research_model = (
        configurable_model.bind_tools(tools)
        .with_retry()
        .with_config(research_model_config)
    )

    researcher_messages = state.get("researcher_messages", [])
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)

    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
        },
    )


async def execute_tool_safely(tool, args, config):
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearchState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    configurables = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]

    has_tool_calls = bool(most_recent_message.tool_calls)
    if not has_tool_calls:
        return Command(goto="compress_research")

    tools = {
        "tavily_search": tavily_search,
        "think_tool": think_tool,
        "ResearchComplete": ResearchComplete,
    }

    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools[tc["name"]], tc["args"], config) for tc in tool_calls if tc["name"] in tools
    ]
    observations = await asyncio.gather(*tool_execution_tasks)

    tool_outputs = [
        ToolMessage(content=obs, name=tc["name"], tool_call_id=tc["id"])
        for obs, tc in zip(observations, tool_calls)
    ]

    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurables.max_react_tool_calls
    research_complete_called = any(tc["name"] == "ResearchComplete" for tc in most_recent_message.tool_calls)

    if exceeded_iterations or research_complete_called:
        return Command(goto="compress_research", update={"researcher_messages": tool_outputs})

    return Command(goto="researcher", update={"researcher_messages": tool_outputs})


async def compress_research(state: ResearchState, config: RunnableConfig):
    configurables = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config(
        {
            "model": configurables.compression_model,
            "max_tokens": configurables.compression_model_max_tokens,
            "api_key": get_api_key_for_model(configurables.compression_model, config),
            "tags": ["langsmith:nostream"],
        }
    )

    researcher_messages = state.get("researcher_messages", [])
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))

    compression_prompt = compress_research_system_prompt.format(date=get_today_date())
    messages = [SystemMessage(content=compression_prompt)] + researcher_messages
    try:
        response = await synthesizer_model.ainvoke(messages)
        raw_notes_content = "\n".join(
            [str(msg.content) for msg in filter_messages(researcher_messages, include_types=["tool", "ai"])]
        )
        return {"compressed_research": str(response.content), "raw_notes": [raw_notes_content]}
    except Exception:
        raw_notes_content = "\n".join(
            [str(msg.content) for msg in filter_messages(researcher_messages, include_types=["tool", "ai"])]
        )
        return {
            "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
            "raw_notes": [raw_notes_content],
        }


researcher_builder = StateGraph(ResearchState, output=ResearcherOutputState, config_schema=Configuration)
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

    configurables = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurables.final_report_model,
        "max_tokens": configurables.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurables.final_report_model, config),
        "tags": ["langsmith:nostream"],
    }

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
        return {
            "final_report": f"Error generating final report: {e}",
            "messages": [AIMessage(content="Report generation failed due to an error")],
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
