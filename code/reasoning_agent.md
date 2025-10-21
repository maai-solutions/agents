# `ReasoningAgent` – Overview

The **`ReasoningAgent`** class (found in `src/linus/agents/agent/reasoning_agent.py`) is the core “two‑call” agent used by the Linus framework when the underlying LLM does **not** support tool‑calling natively (e.g., Gemma 3 27B). It implements an iterative **reasoning → execution → completion‑check** loop that can:

1. **Analyse** a user request and produce a structured plan (JSON) describing the required steps and which tools, if any, should be used.  
2. **Generate arguments** for each planned tool, invoke the tool, and collect its results.  
3. **Validate** whether the overall task is complete, possibly iterating up to `max_iterations` times.

The agent is heavily instrumented with telemetry, rich‑console logging, and optional memory integration for context‑aware reasoning.

---

## Key Responsibilities

| Responsibility | How it is implemented |
|----------------|-----------------------|
| **Prompt engineering** | Three private prompt generators (`_create_reasoning_prompt`, `_create_execution_prompt`, `_create_completion_check_prompt`) build system‑level prompts that embed available tool descriptions and guide the LLM to return strict JSON. |
| **Reasoning phase** | `_reasoning_call` sends the *reasoning prompt* + user input to the LLM, parses the JSON response into a `ReasoningResult` (contains `has_sufficient_info`, `reasoning` text, and a list of planned tasks). |
| **Execution phase** | For each task: <br>• If a `tool_name` is provided, `_generate_tool_arguments` creates JSON arguments using the *execution prompt*.<br>• The tool is looked up in `self.tool_map` and executed via `await tool.arun(args)`. <br>• If no tool is needed, `_generate_response` directly asks the LLM for a response. |
| **Completion check** | `_check_completion` builds a prompt with the original request and a summary of execution history, asks the LLM to answer a JSON‑structured “is it complete?” question, and parses the result. |
| **Memory integration** | Optional `MemoryManager` can be supplied. The agent stores the user prompt, intermediate results, and the final assistant answer as `MemoryEntry` objects, and can prepend a token‑budget‑aware memory context to each iteration. |
| **Telemetry & metrics** | Every LLM call, tool execution, and completion check is wrapped in `self.telemetry` spans. Metrics (`AgentMetrics`) are updated throughout the run (iterations, token usage, tool success/failure, etc.). |
| **Rich console output** | When the optional `rich` library is available, the agent renders tables and panels for metrics, reasoning, and execution progress. |
| **Configurable parameters** | The constructor accepts many knobs (temperature, max_tokens, top_p/k, memory ratios, LLM config, etc.) and falls back to sensible defaults. |

---

## Core Public API

```python
class ReasoningAgent(Agent):
    async def run(
        self,
        input_data: Union[str, BaseModel, Dict[str, Any]],
        return_metrics: bool = True
    ) -> Union[str, BaseModel, AgentResponse]:
        """Execute the full reasoning‑execution loop."""
```

*`run`* is the entry point used by higher‑level orchestration code. It returns either a plain result string (if `return_metrics=False`) or an `AgentResponse` object containing:

- `result` – the final formatted answer.  
- `metrics` – an `AgentMetrics` instance with detailed execution stats.  
- `execution_history` – list of all tasks, tool names, results, and statuses.  
- `completion_status` – the JSON payload from the completion‑check phase.

---

## Important Private Helpers

| Method | Purpose |
|--------|---------|
| `_create_reasoning_prompt` | Builds the system prompt that lists available tools and asks the model to output a JSON plan. |
| `_create_execution_prompt` | Template used to ask the model to generate arguments for a specific tool. |
| `_create_completion_check_prompt` | Template used to ask the model whether the overall task is done. |
| `_reasoning_call` | Sends the reasoning prompt, parses JSON, and records telemetry. |
| `_generate_tool_arguments` | Calls the LLM with the execution prompt and extracts JSON arguments (handles markdown code blocks, raw JSON, etc.). |
| `_execute_task_with_tool` | Looks up the tool, runs it, captures results, and updates telemetry/metrics. |
| `_generate_response` | Direct LLM response for tasks that don’t need a tool. |
| `_check_completion` | Summarises execution history and asks the model for a completion decision. |
| `_format_final_response_with_history` | Synthesises all findings into a coherent final answer using the LLM. |
| `_display_metrics_rich` / `_display_reasoning_rich` | Optional pretty‑printing when `rich` is installed. |

---

## Typical Execution Flow (simplified)

1. **Input validation** – `run` normalises `input_data` to a string.  
2. **Telemetry start** – `trace_agent_run` span opens.  
3. **Memory store (optional)** – user prompt added to memory.  
4. **Loop (≤ `max_iterations`)**  
   - Build **context**: optional memory slice + shared state + prior execution history.  
   - **Reasoning** → obtain JSON plan.  
   - **Execution** → for each task: generate tool args → run tool (or LLM response).  
   - Append results to `execution_history`.  
   - **Completion check** → ask LLM if done. If `is_complete=True` break; else continue.  
5. **Finalize metrics**, store final assistant output in memory, log telemetry.  
6. **Return** – either plain result or full `AgentResponse`.

---

## Extensibility Points

- **Tool set** – Pass any list of `BaseTool` subclasses; the reasoning prompt automatically enumerates them.  
- **Memory backend** – Swap `MemoryBackend` implementations (in‑memory, vector store, custom) via `MemoryManager`.  
- **Telemetry provider** – The DI‑based `ITelemetry` implementation can be replaced (e.g., Langfuse, OpenTelemetry).  
- **Prompt customisation** – Subclass `ReasoningAgent` and override `_create_*_prompt` methods for domain‑specific wording.  
- **Parameter overrides** – All LLM generation knobs (`temperature`, `max_tokens`, `top_p/k`) can be supplied at construction or via `LLMConfig`.

---

## Summary

`ReasoningAgent` is a **self‑contained orchestration layer** that enables agents to work with LLMs lacking native tool calling. By separating **planning**, **tool argument generation**, **tool execution**, and **completion validation**, it provides a deterministic, observable, and configurable workflow suitable for complex multi‑step tasks while preserving token‑budget awareness through optional memory integration.

---
