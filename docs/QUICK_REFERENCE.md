# CoordinatorAgent Infinite Loop - Quick Reference Guide

## Problem Summary
After Researcher subagent executes its first search, CoordinatorAgent enters an infinite loop due to:
- Context rebuilding with original request every iteration
- Full re-planning from scratch each iteration  
- No safeguard against perpetual "continue" decisions
- Subagents receiving full original request instead of just their step

---

## Critical Code Sections

### 1. Main Loop (Where the Problem Manifests)
**File**: `src/linus/agents/agent/coordinator_agent.py`
**Lines**: 266-406 (`_run_with_trace` method)

**Key Issue**: Lines 296-344
```python
while not is_complete and iteration < self.max_iterations:
    # Context rebuilt fresh EVERY iteration
    context = self._build_context(input_text, execution_history, current_plan)
    
    # Full plan re-created EVERY iteration
    current_plan = await self._create_plan(context, iteration)
    
    # Execute and extend history
    step_results = await self._execute_plan(current_plan, execution_history, input_text)
    execution_history.extend(step_results)  # <- Grows unbounded
    
    # Evaluate - but NO safeguard for "continue" counter
    evaluation_result = await self._evaluate_progress(...)
    
    # No counter for consecutive "continue" responses
    if evaluation_result["next_action"] == "continue":
        is_complete = evaluation_result["task_completed"]
```

---

### 2. Context Building (Re-adds Original Request)
**File**: `src/linus/agents/agent/coordinator_agent.py`
**Lines**: 408-442 (`_build_context` method)

**Key Problem**: Line 415
```python
context = input_text  # <- Original request always included

# Even after execution, context still includes the full original request
# This causes LLM to re-read and potentially re-plan the entire task
```

**Impact**: 
- Iteration 1: Context = "Find info, calculate, explain"
- Iteration 2: Context = "Find info, calculate, explain" + results
- LLM re-reads original request again → might re-plan same steps

**Fix**: 
```python
# Instead of including full input_text, only include step count
# or change prompt to be incremental: "what's next?" not "create full plan"
```

---

### 3. Planning (Full Re-Planning Each Iteration)
**File**: `src/linus/agents/agent/coordinator_agent.py`
**Lines**: 444-494 (`_create_plan` method)

**Problem**: Every iteration calls `_create_plan` asking "What's the full plan?"

**Compare to ReasoningAgent** (works better):
- `reasoning_agent.py` lines 557-671
- Asks: "What tasks remain?" (incremental)
- Not: "What's the full plan?" (full re-plan)

**Fix**: Change planning prompt to ask for NEXT steps, not full plan

---

### 4. Subagent Context (Includes Full Original Request)
**File**: `src/linus/agents/agent/coordinator_agent.py`
**Lines**: 583-587 (`_execute_step` method)

**Problem Code**:
```python
enriched_input = f"""Original request: {original_request}

Current step: {step['description']}

{step_input}"""
```

**Why It's Wrong**:
- Subagent sees full original request (e.g., "Find info, calculate, explain")
- But this step is only "Find info"
- Subagent (a ReasoningAgent) might try to handle all 3 tasks
- Can cause scope creep and internal looping

**Fix**:
```python
# Don't pass original_request, only pass step-specific context
enriched_input = f"""Current task: {step['description']}

{step_input}

Previous results: {prev_results}"""
```

---

### 5. Evaluation (No Safeguard)
**File**: `src/linus/agents/agent/coordinator_agent.py`
**Lines**: 624-693 (`_evaluate_progress` method)

**Missing Safeguard**:
```python
# No counter for consecutive "continue" responses
# No counter for consecutive "replan" responses
# If evaluation says "continue" 3+ times, should force complete
```

**Compare to ReasoningAgent** (has safeguard):
**File**: `src/linus/agents/agent/reasoning_agent.py`
**Lines**: 883-894 (`_check_completion` method)

```python
# Check for repetitive tool calls - force completion if same tool called 3+ times
if len(execution_history) >= 3:
    recent_tools = [item.get('tool') for item in execution_history[-3:]]
    if len(set(recent_tools)) == 1 and recent_tools[0] is not None:
        # Force completion due to repetitive behavior
        return {"is_complete": True, ...}
```

---

### 6. Plan Execution (Skip Logic Might Fail)
**File**: `src/linus/agents/agent/coordinator_agent.py`
**Lines**: 496-542 (`_execute_plan` method)

**Potential Issue** (lines 513-520):
```python
completed_steps = {item["step_number"] for item in execution_history}

for step in plan["plan"]:
    step_number = step["step_number"]
    
    # If plan is re-created with different step_numbers
    # This skip logic will fail
    if step_number in completed_steps:
        continue
```

**Why It Fails**:
- Iteration 1 Plan: Step 1, 2, 3
- Iteration 2 Plan: Step 1, 2, 3, 4 (different!)
- completed_steps = {1, 2, 3}
- New Step 4 gets executed
- Loop might continue

---

## The Infinite Loop Sequence

```
1. Researcher completes (Step 1)
   ├─ execution_history = [Step 1 result]
   └─ evaluation = "not_complete, continue"

2. Context rebuilt with original request + history
   ├─ _create_plan() called
   └─ New plan might have different structure

3. Step skip logic fails (if step_numbers changed)
   ├─ Steps re-executed or new steps added
   └─ execution_history grows

4. Evaluation says "continue" again
   └─ Loop continues (no safeguard for repeated "continue")

5. Repeat until max_iterations reached
```

---

## Quick Fixes (Estimated Time & Impact)

| Fix | Time | Impact | Difficulty |
|-----|------|--------|------------|
| Add counter for consecutive "continue" | 5 min | Stops most infinite loops | Very Easy |
| Remove original_request from subagent input | 10 min | Reduces scope creep | Easy |
| Limit history in context to last 3-5 items | 5 min | Reduces re-planning noise | Easy |
| Add unique step IDs instead of step_number | 15 min | Prevents skip logic failures | Medium |
| Change planning prompt to incremental | 15 min | Better alignment with task | Medium |
| Add "replan" counter safeguard | 5 min | Prevents replan loops | Very Easy |

---

## Files to Modify

1. **Primary**: `/Users/udg/Projects/ai/agents/src/linus/agents/agent/coordinator_agent.py`
   - Main loop (lines 296-344)
   - Context building (lines 408-442)
   - Subagent input (lines 583-587)
   - Evaluation handling (lines 333-340)

2. **Reference**: `src/linus/agents/agent/reasoning_agent.py`
   - See how ReasoningAgent handles safeguards (lines 883-894)
   - See how it builds context (lines 276-300)

3. **Config**: `src/linus/agents/agent/models.py`
   - AgentMetrics definition (lines 27-61)

---

## Testing

After fixes, test with:
```python
# Simple test - should complete in 1 iteration
coordinator.run("What is 2+2?")

# Multi-step test - should complete in 2-3 iterations max
coordinator.run("Search for Python, calculate 100*5, explain results")

# Should NOT infinite loop even with ambiguous criteria
coordinator.run("Find information and do something with it")
```

---

## Related Issues

- The coordinator was recently added (commit `fd274ba`)
- Used in: `/Users/udg/Projects/ai/agents/src/app.py`
- Example: `/Users/udg/Projects/ai/agents/examples/example_coordinator_usage.py`
- This is likely a new feature not fully tested

---

## Summary

**Root Cause**: The plan-execute-evaluate loop rebuilds context with the original request every iteration and re-plans everything from scratch. Without a safeguard against perpetual "continue" decisions, it can loop indefinitely.

**Most Likely Fix**: Add a counter for consecutive "continue" responses and force completion after 3+ consecutive "continue"s, similar to ReasoningAgent's safeguard.

