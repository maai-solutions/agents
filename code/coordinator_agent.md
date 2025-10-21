# CoordinatorAgent Overview

The **`CoordinatorAgent`** (defined in `src/linus/agents/agent/coordinator_agent.py`) is a high‑level orchestrator that manages a collection of **sub‑agents** to solve complex user requests. It follows a **plan‑execute‑evaluate** loop, repeatedly generating a plan, running the assigned sub‑agents, and evaluating progress until the task is complete or a safety limit is reached.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **SubAgent** | Lightweight wrapper around an individual `Agent` instance, storing its `name`, `description`, and list of `capabilities`. |
| **Coordinator Loop** | Repeats up to `max_iterations` (default 15): <br>1️⃣ Build context (user request, memory, state, execution history) <br>2️⃣ **Plan** – ask the LLM to produce a JSON‑structured high‑level plan. <br>3️⃣ **Execute** – run each step with the appropriate sub‑agent, handling dependencies and storing results in shared state. <br>4️⃣ **Evaluate** – ask the LLM to decide whether the plan is still valid, if the task is finished, or if replanning is required. |
| **SharedState** | A key/value store (`state`) that sub‑agents can read/write, enabling later steps to use results from earlier steps. |
| **MemoryManager** (optional) | Provides long‑term context. A configurable fraction (`memory_context_ratio`) of the available memory window is injected into the planning context. |
| **Telemetry / Tracing** | All major actions (`run`, `plan`, `execute_step`, `evaluate`) are wrapped with telemetry traces, enabling hierarchical observability. |
| **Rich Logging** | If the `rich` library is available, the coordinator prints pretty‑formatted plans and metrics in the console. |

## Core Workflow

1. **Initialization (`__init__`)**  
   - Receives an OpenAI client, model name, a list of `SubAgent` objects, optional tools, and many configuration knobs (temperature, max tokens, etc.).  
   - Sets up internal maps (`self.subagent_map`), generation kwargs, and creates two prompt templates: `self.planning_prompt` and `self.evaluation_prompt`.

2. **Running the Agent (`run`)**  
   - Validates and normalises the input, then delegates to `_run_with_trace` inside a telemetry span.

3. **Main Loop (`_run_with_trace`)**  
   - Creates a fresh `AgentMetrics` object to record iterations, token usage, and execution time.  
   - Stores the user request in memory (if a `MemoryManager` is provided).  
   - Repeats the **plan‑execute‑evaluate** cycle:
     - **Planning** (`_create_plan`) – builds a context string via `_build_context` and calls the LLM with `self.planning_prompt`. The response is parsed as JSON and becomes `current_plan`.
     - **Execution** (`_execute_plan`) – iterates through each step in `current_plan["plan"]`.  
       * Skips already‑completed steps.  
       * Checks declared dependencies.  
       * Calls `_execute_step` which invokes the assigned sub‑agent’s `run` method, captures its result, stores it in `SharedState`, and records the step outcome.
     - **Evaluation** (`_evaluate_progress`) – assembles a concise summary of the plan and execution history, sends it to the LLM using `self.evaluation_prompt`, and parses the JSON response. The response tells the coordinator whether to **continue**, **replan**, or **complete**, plus optional suggested changes.
     - The loop terminates when the LLM signals completion, when the maximum iteration count is reached, or when safety counters (3 consecutive `continue`/`replan` without progress) force termination.

4. **Result Formatting (`_format_final_response`)**  
   - If the task succeeded, aggregates the full results (preferring the stored full result from `SharedState`) and may invoke the LLM again to synthesize a coherent final answer.

5. **Metrics & Display**  
   - After the loop, execution metrics (iterations, token usage, elapsed time, task‑completed flag) are logged and optionally rendered with `rich`.  
   - Metrics are sent to the telemetry system.

## Important Helper Methods

| Method | Purpose |
|--------|---------|
| `_build_context` | Constructs the prompt context, injecting memory snippets, shared state, and recent execution history. |
| `_create_plan` | Sends the planning prompt to the LLM and parses the JSON plan. |
| `_execute_plan` | Drives step‑wise execution, handling dependencies and state updates. |
| `_execute_step` | Calls a specific sub‑agent, builds enriched input (including prior step results), and records the outcome. |
| `_evaluate_progress` | Sends the evaluation prompt to the LLM and parses its decision JSON. |
| `_format_final_response` | Produces a user‑facing answer from the execution history, optionally using the LLM for synthesis. |
| `_display_plan_rich` / `_display_metrics_rich` | Pretty‑print the plan and metrics when `rich` is installed. |

## How the Coordinator Handles Edge Cases

* **Missing Sub‑Agent** – Returns a failure result for that step.  
* **Dependency Failure** – Skips the step and records a “dependencies not met” message.  
* **LLM Parsing Errors** – Logs the exception and falls back to a minimal JSON structure indicating continuation.  
* **Stuck Loop** – Counters (`consecutive_continues`, `consecutive_replans`) trigger forced completion after three idle iterations.  
* **Memory & State Limits** – `memory_context_ratio` caps how much of the memory window is injected; `SharedState` respects a configurable token budget.

## Summary

`CoordinatorAgent` is a robust orchestration layer that:

* Translates a high‑level user request into a concrete, ordered plan.  
* Delegates each plan step to the most suitable specialized sub‑agent.  
* Continuously evaluates progress with LLM‑driven reasoning, allowing dynamic replanning.  
* Maintains observability via telemetry, optional rich console output, and detailed metrics.

It enables complex, multi‑tool workflows to be executed reliably without the caller having to manage the intricate sequencing logic.
