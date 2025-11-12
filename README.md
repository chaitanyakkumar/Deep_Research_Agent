# Deep Researcher — Multi-Agent Research Workflow (LangGraph + LangChain)

## Overview
The **Deep Researcher** project implements a fully modular, asynchronous, multi-agent workflow built using **LangGraph** and **LangChain**.  
It automates end-to-end research — from clarifying a user query to generating a polished final report — using structured nodes and configurable models.

---

## Workflow Architecture

### Clarify With User
- Parses user input and checks if the query needs clarification.  
- If needed, the agent asks follow-up questions before continuing.  
- Uses `UserQueryState` structured output for controlled clarification.

**Node:** `clarify_with_user`  
**Next:** → `write_research_brief`

---

### 2️⃣ Write Research Brief
- Converts clarified user queries into a structured **research brief**.  
- Defines clear scope, objectives, and research constraints.  
- Sets up a `Supervisor` context to guide the main research phase.

**Node:** `write_research_brief`  
**Next:** → `research_supervisor`

---

###  Supervisor Agent
- Acts as the **lead researcher**, breaking down the brief into subtasks.  
- Delegates research to multiple **Researcher** agents in parallel.  
- Uses 3 key tools:
  - 🧩 `ConductResearch` – spawns subgraph researchers  
  - ✅ `ResearchComplete` – signals completion  
  - 💭 `think_tool` – internal reasoning & planning  

**Subgraph:** `supervisor_subgraph`  
**Next:** → `supervisor_tools`

---

### Researcher Subgraph
Each **Researcher** agent:
- Performs focused topic research using available tools (`tavily_search`, `think_tool`, etc.)
- Gathers findings via API tools or native web search
- Iterates and refines until complete
- Compresses results into concise structured summaries

**Nodes:**
- `researcher`
- `researcher_tools`
- `compress_research`

**Output:** `compressed_research` + raw notes

---

### Final Report Generation
- Synthesizes all notes and compressed research into a polished **final report**.  
- Uses token-limit–aware retries for long documents.  
- Produces an executive-style summary with supporting details.

**Node:** `final_report_generation`  
**Next:** → `END`

---

## Credits
- This implementation was inspired and learned from the official [LangGraph Deep Research repository][https://github.com/langchain-ai/open_deep_research]
- I plan to extend the project by enabling scientific resaerch, acquiring knowledge from pdfs, adding tools with MCP and adding memory based on requirements
- Huge thanks to the LangGraph and LangChain teams for their incredible open-source work that made this project possible.
