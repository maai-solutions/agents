# CoordinatorAgent Infinite Loop Investigation - Final Report

## Summary

The CoordinatorAgent implements a **plan-execute-evaluate loop** that can enter an infinite loop after the Researcher subagent completes its first search. The root causes are:

1. **Context always includes the full original request**, causing the LLM to re-reason about the entire task in each iteration
2. **Planning is done from scratch every iteration** instead of incrementally planning the next steps
3. **No safeguard against perpetual "continue" or "replan" decisions** from the evaluation phase
4. **Subagents receive the full original request**, not just their assigned step context

---

## Architecture: The Plan-Execute-Evaluate Loop

### High-Level Flow

```
Coordinator.run(input)
  ├─> Loop (max 15 iterations):
  │    ├─ Build Context (includes original request + history)
  │    ├─ Create Plan (LLM re-plans everything)
  │    ├─ Execute Plan (runs steps, skipping completed ones)
  │    ├─ Evaluate Progress (decides: complete/replan/continue)
  │    └─ If not complete, continue loop
  └─> Return result
```

### File Locations

- **Main Implementation**: `/Users/udg/Projects/ai/agents/src/linus/agents/agent/coordinator_agent.py`
- **Base Class**: `/Users/udg/Projects/ai/agents/src/linus/agents/agent/base.py`
- **Factory Functions**: `/Users/udg/Projects/ai/agents/src/linus/agents/agent/factory.py`
- **Example Usage**: `/Users/udg/Projects/ai/agents/examples/example_coordinator_usage.py`

---

## Root Cause #1: Context Rebuilding with Original Request

**Location**: `coordinator_agent.py`, `_build_context()` method, lines 408-442

**Problem Code**:
```python
def _build_context(self, input_text, execution_history, current_plan):
    context = input_text  # <- Original request ALWAYS included
    
    if self.memory_manager:
        memory_context = self.memory_manager.get_context(...)
        context = f"{memory_context}\n\n=== Current Task ===\n{context}"
    
    state_data = self.state.get_all()
    if state_data:
        state_context = f"\n\nShared state: {json.dumps(...)}"
        context = context + state_context
    
    # Add execution history from LAST 10 items
    if execution_history:
        history_context = "\n\n=== Previous Execution Results ===\n" + "\n".join([
            f"- Step {item['step_number']}: {item['subagent']} - {item['result'][:200]}"
            for item in execution_history[-10:]  # <- Only last 10
        ])
        context = context + history_context
    
    return context
```

**Impact**:
- In iteration 1: Context = "Find info about X, calculate Y, explain Z"
- In iteration 2: Context = "Find info about X, calculate Y, explain Z" + previous results
- LLM in iteration 2 re-reads the original request AGAIN
- This can cause re-planning of steps already completed

---

## Root Cause #2: Full Re-Planning Every Iteration

**Location**: `coordinator_agent.py`, main loop in `_run_with_trace()`, lines 296-344

**Problem Code**:
```python
while not is_complete and iteration < self.max_iterations:
    iteration += 1
    
    # Build fresh context
    context = self._build_context(input_text, execution_history, current_plan)
    
    # PROBLEM: Creates complete plan from scratch every iteration
    # Instead of: "What's next?" asks "What's the full plan?"
    current_plan = await self._create_plan(context, iteration)
    
    if not current_plan or not current_plan.get("plan"):
        break
    
    # Execute plan steps (skipping completed ones)
    step_results = await self._execute_plan(current_plan, execution_history, input_text)
    execution_history.extend(step_results)  # <- Grows unbounded
    
    # Evaluate if complete
    evaluation_result = await self._evaluate_progress(
        input_text, current_plan, execution_history
    )
    
    # Determine next action
    if evaluation_result["next_action"] == "complete":
        is_complete = True
    elif evaluation_result["next_action"] == "replan":
        # Loop continues, context gets rebuilt with history
        pass
    elif evaluation_result["next_action"] == "continue":
        is_complete = evaluation_result["task_completed"]
    
    # No safeguard! If LLM says "continue" 3+ times in a row, keeps looping
```

**Impact**:
- Iteration 1: Plan = [Step 1: Research, Step 2: Calculate, Step 3: Explain]
- Iteration 2: Plan = [Step 1: Research, Step 2: Calculate, Step 3: Explain, Step 4: ...]?
- LLM might create different plans each time
- Skip logic (based on step_number) might fail if plans differ

---

## Root Cause #3: Subagents Receive Full Original Request

**Location**: `coordinator_agent.py`, `_execute_step()` method, lines 544-622

**Problem Code**:
```python
async def _execute_step(self, step, execution_history, original_request):
    step_number = step["step_number"]
    subagent_name = step["assigned_subagent"]
    step_input = step["input"]
    
    # Prepare enriched input - BUT includes full original request!
    enriched_input = f"""Original request: {original_request}

Current step: {step['description']}

{step_input}"""
    
    # Add previous results (last 3 items only)
    if execution_history:
        prev_results = "\n".join([
            f"- Step {item['step_number']}: {item['result'][:150]}"
            for item in execution_history[-3:]
        ])
        enriched_input += f"\n\nPrevious results:\n{prev_results}"
    
    # PROBLEM: Subagent is a ReasoningAgent with max_iterations=10
    # It sees: "Find info about Python AND calculate compound interest AND explain"
    # Even though this step is only "Research Python"
    result = await subagent.agent.run(enriched_input, return_metrics=False)
```

**Concrete Example**:

Iteration 1, Step 1 execution:
```
enriched_input = """Original request: Find information about Python, 
                     calculate compound interest, and explain

Current step: Research and summarize information about Python

Please search for and provide comprehensive information about Python"""

Result: Researcher returns "Python is..."
```

Iteration 2, Step 3 execution:
```
enriched_input = """Original request: Find information about Python, 
                     calculate compound interest, and explain

Current step: Explain all results

Previous results:
- Step 1: Python is...
- Step 2: 50000 * 1.07^5 = 70128.80

Please explain the results"""

Result: General agent sees full original request AGAIN
```

**Impact**:
- Subagent sees the full scope, not just their specific task
- A ReasoningAgent with max_iterations=10 will try to handle all 3 tasks
- This can expand the scope beyond what was intended

---

## Root Cause #4: No Safeguard Against Perpetual "Continue" Decisions

**Location**: `coordinator_agent.py`, main loop `_run_with_trace()`, lines 333-340

**Problem Code**:
```python
# Determine next action
if evaluation_result["next_action"] == "complete":
    is_complete = True
elif evaluation_result["next_action"] == "replan":
    # No counter - loop continues
    self.logger.info("[COORDINATOR] Replanning based on evaluation")
elif evaluation_result["next_action"] == "continue":
    # No counter - just sets is_complete based on task_completed
    is_complete = evaluation_result["task_completed"]

if not is_complete and iteration >= self.max_iterations:
    self.logger.warning(f"[COORDINATOR] Max iterations reached")
    break
```

**Missing**:
- No counter for consecutive "continue" responses
- No counter for consecutive "replan" responses
- If evaluation says "continue" for 3+ iterations, should force complete
- Comparison: ReasoningAgent has safeguard at line 883-894 in reasoning_agent.py

---

## Comparison: Why ReasoningAgent Works Better

### ReasoningAgent Loop (Works)
**File**: `reasoning_agent.py`, lines 267-371

```python
while not is_complete and iteration < self.max_iterations:
    # Build context with history
    context = input_text + memory + state + execution_history[-5:]  # <- Only last 5
    
    # Key difference: Asks "what tasks remain?" not "what's the full plan?"
    reasoning_result = await self._reasoning_call(context, iteration)
    
    # Execute tasks for this iteration
    for task in reasoning_result.tasks:
        if task.tool_name:
            result = await self._execute_task_with_tool(task, context)
        else:
            result = await self._generate_response(task.description, context)
    
    # KEY: Has safeguard against repetitive tool calls
    completion_status = await self._check_completion(input_text, execution_history)
    is_complete = completion_status["is_complete"]
```

**Safeguard** (lines 883-894):
```python
async def _check_completion(self, original_request, execution_history):
    # Check for repetitive tool calls - force completion if same tool called 3+ times
    if len(execution_history) >= 3:
        recent_tools = [item.get('tool') for item in execution_history[-3:]]
        if len(set(recent_tools)) == 1 and recent_tools[0] is not None:
            self.logger.warning(f"[ASYNC-COMPLETION] Detected repetitive tool calls")
            return {
                "is_complete": True,
                "reasoning": "Forced completion due to repetitive tool usage",
                "next_action": "none",
                "missing_steps": []
            }
```

### CoordinatorAgent Loop (Problems)

**Missing**:
1. No query for "what's next?" - re-plans everything
2. No safeguard for repetitive decisions
3. Context includes full original request every iteration
4. No "next_action" override for repetitive behavior

---

## Concrete Execution Scenario

### Why It Loops After Researcher First Search

1. **Iteration 1, Step 1**: Researcher executes successfully
   - Input: "Find information about X" (from enriched_input)
   - Researcher's internal loop (ReasoningAgent) runs 10 times max, completes after 2-3
   - Returns results
   - History: [Step 1: completed, "Found X information"]

2. **Iteration 1, Step 2**: Calculator executes successfully
   - Input: "Calculate Y" + "Previous: Found X information"
   - Calculator's internal loop completes
   - History: [Step 1: ..., Step 2: completed, "Y = result"]

3. **Iteration 1, Step 3**: Skipped (no dependencies met)
   - evaluation_result = "task_completed=false, next_action=continue"
   - (LLM sees Plan with 3 steps but only 2 completed)

4. **Iteration 2** - LOOP CONTINUES:
   - Context rebuilt: "Find information about X, calculate Y, explain Z" + previous results
   - _create_plan() called again
   - LLM might plan: [Step 1, 2, 3] (same) or [Step 1, 2, 3, 4] (different)
   - If same step_numbers: Step 1 and 2 skipped, Step 3 executed
   - If different step_numbers: Skip logic fails, might re-execute

5. **Iteration 3+**: If evaluation keeps saying "continue", loop persists

---

## Visual: The Context Spiral

```
Iteration 1:
  Context = "Find info... calculate... explain..."
  Plan = [Step 1, 2, 3]
  Execute: Steps 1, 2
  Evaluation: task_completed=false (Step 3 pending)
  
Iteration 2:
  Context = "Find info... calculate... explain..." + [Step 1 results, Step 2 results]
  Plan = [Step 1, 2, 3] or [Step 1, 2, 3, 4, ...]  <- UNCERTAIN
  Execute: Skip 1, 2, run 3 (if same steps) OR might run different steps
  Evaluation: task_completed=true OR task_completed=false (depends on plan)
  
Iteration 3:
  If evaluation said task_completed=false:
  Context = "Find info... calculate... explain..." + [all previous results]
  Plan = DIFFERENT PLAN? Same plan?
  Execute: ???
  
Iteration 4-15:
  Loop may repeat indefinitely until:
  - max_iterations reached, OR
  - Evaluation says "task_completed=true", OR
  - Some other break condition triggered
```

---

## Why This Affects Researcher First

After Researcher (Step 1) completes:
1. The execution history shows "Step 1: completed"
2. The coordinator re-plans from scratch with this history
3. The new plan might assign Step 2, 3, 4, ...
4. But the evaluation logic might not clear

The infinite loop is **most likely** if:
- The evaluation prompt is ambiguous about completion criteria
- The LLM evaluator keeps saying "need more steps" even though original task is satisfied
- New steps are created in each iteration
- Or steps are re-executed due to step_number mismatches

---

## Impact on Development Status

**Current Commit**: `fd274ba coordinator agent` (recent change)

The CoordinatorAgent was recently added and may have issues that weren't caught:
- Used in: `/Users/udg/Projects/ai/agents/src/app.py`
- Example: `/Users/udg/Projects/ai/agents/examples/example_coordinator_usage.py`
- Factory: `/Users/udg/Projects/ai/agents/src/linus/agents/agent/factory.py` (Coordinator function)

---

## Recommended Fixes (Priority Order)

### Quick Fixes (No Major Refactoring)

1. **Add "continue" counter** (5 min)
   - Track consecutive "continue" responses
   - Force `is_complete=true` after 3 consecutive "continue"s

2. **Change step input to subagent** (10 min)
   - Don't pass `original_request`, only step description
   - Keep subagent focused on its assigned task

3. **Add safeguard for same step_number re-execution** (10 min)
   - Track executed step IDs, not just step numbers
   - Generate unique IDs if step_numbers might repeat

### Medium Fixes (Some Refactoring)

4. **Change planning prompt** (15 min)
   - From: "Create a plan to accomplish X"
   - To: "Given we've completed X and Y, what's the next step?"

5. **Limit history in context** (10 min)
   - Only include last 3-5 execution results, not last 10
   - Include count of total steps completed, not all details

### Major Fixes (Architectural)

6. **Change from "replan each iteration" to "incrementally plan"**
   - Track overall strategy
   - Plan next steps, not re-plan everything

---

## Test Cases to Verify

```python
# Should complete in 2-3 iterations
coordinator.run("Search for Python, calculate 100*5, explain")

# Should complete immediately
coordinator.run("Calculate 2+2")

# Should complete in 2 iterations max
coordinator.run("Search for X then explain")
```

