# CoordinatorAgent Citation Support Fix

## Problem

Citations were not appearing in API responses when using the CoordinatorAgent with subagents. The issue was that:

1. CoordinatorAgent was calling subagents with `return_metrics=False`
2. This only returned the result string, not the full `AgentResponse` with citations
3. Citations from subagents were being lost

## Solution

Updated the CoordinatorAgent to:

1. **Request full AgentResponse from subagents** - Changed `return_metrics=False` to `return_metrics=True`
2. **Extract citations from subagent responses** - Parse AgentResponse and collect citations
3. **Aggregate citations from all steps** - Collect citations across all subagent executions
4. **Include citations in final response** - Pass citations to the CoordinatorAgent's AgentResponse

## Changes Made

### File: `src/linus/agents/agent/coordinator_agent.py`

#### 1. Import Citation Model (Line 14)
```python
from .models import AgentMetrics, AgentResponse, Citation
```

#### 2. Initialize Citations Collection (Line 290)
```python
# Track execution history
execution_history = []
citations = []  # Collect citations from subagents
```

#### 3. Update `_execute_step` to Get Full Response (Lines 700-731)
```python
# Get full AgentResponse to collect citations
result = await subagent.agent.run(enriched_input, return_metrics=True)

# Extract result text and citations
if isinstance(result, AgentResponse):
    result_text = str(result.result)
    step_citations = result.citations if result.citations else []
else:
    result_text = str(result)
    step_citations = []

return {
    "step_number": step_number,
    "subagent": subagent_name,
    "description": step["description"],
    "status": "completed",
    "result": result_text,
    "citations": step_citations  # Include citations in step result
}
```

#### 4. Collect Citations from Step Results (Lines 320-326)
```python
# Add step results to history and collect citations
for step_result in step_results:
    execution_history.append(step_result)
    # Collect citations from this step
    if "citations" in step_result and step_result["citations"]:
        citations.extend(step_result["citations"])
        self.logger.debug(f"[COORDINATOR] Collected {len(step_result['citations'])} citations from step {step_result['step_number']}")
```

#### 5. Include Citations in Final Response (Lines 437-442)
```python
# Log total citations collected
if citations:
    self.logger.info(f"[COORDINATOR] Collected {len(citations)} total citations from subagents")

return AgentResponse(
    result=formatted_result,
    metrics=metrics,
    execution_history=execution_history,
    completion_status=evaluation_result,
    citations=citations  # Include all collected citations
)
```

## Citation Flow with CoordinatorAgent

```
User Query
    ↓
CoordinatorAgent.run()
    ↓
Creates Plan with Subagent Steps
    ↓
For each step:
    ↓
    Execute Subagent (e.g., Researcher)
        ↓
        Subagent.run(return_metrics=True)
            ↓
            ReasoningAgent uses vector_search tool
                ↓
                VectorStoreTool returns JSON with citations
                    ↓
                ReasoningAgent extracts citations
                    ↓
                Returns AgentResponse with citations
            ↓
        CoordinatorAgent extracts citations from AgentResponse
        CoordinatorAgent adds citations to collection
    ↓
All citations from all steps collected
    ↓
CoordinatorAgent returns AgentResponse with all citations
    ↓
FastAPI endpoint extracts and returns citations
    ↓
User receives response with citations
```

## Testing

### Before Fix
```json
{
  "query": "why are the whales in danger?",
  "response": "...",
  "citations": []  // Empty - citations were lost
}
```

### After Fix
```json
{
  "query": "why are the whales in danger?",
  "response": "...",
  "citations": [
    {
      "document_id": "whale_article_123",
      "chunk_number": 5,
      "score": 0.95,
      "content_preview": "North Atlantic right whale population..."
    },
    {
      "document_id": "conservation_report_456",
      "chunk_number": 12,
      "score": 0.88,
      "content_preview": "Whale populations face threats from..."
    }
  ]
}
```

## Execution History with Citations

Now the execution history includes citations for each step:

```json
{
  "execution_history": [
    {
      "step_number": 1,
      "subagent": "Researcher",
      "description": "Research whale endangerment",
      "status": "completed",
      "result": "Whale populations face...",
      "citations": [
        {
          "document_id": "doc_123",
          "chunk_number": 5,
          "score": 0.95,
          "content_preview": "..."
        }
      ]
    }
  ],
  "citations": [
    // Aggregated citations from all steps
  ]
}
```

## Benefits

1. **Complete Citation Tracking** - Citations from all subagents are preserved
2. **Source Attribution** - Users can see which documents informed the response
3. **Multi-Agent Citations** - Works with complex multi-agent workflows
4. **Backward Compatible** - Subagents without citations still work (empty array)
5. **Debug Logging** - Shows how many citations were collected at each step

## Impact on Performance

Minimal impact:
- Subagents now return `AgentResponse` objects instead of strings
- Small overhead from citation extraction (simple list append operations)
- Citations are already computed by subagents, just preserved instead of discarded

## Verification

Run a query that uses the Researcher subagent with vector search:

```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "why are the whales in danger?",
    "session_id": "test"
  }'
```

Check the response for:
- `citations` array is populated (not empty)
- Each citation has `document_id`, `chunk_number`, `score`, `content_preview`
- Citations correspond to the information in the response

## Related Files

- `src/linus/agents/agent/coordinator_agent.py` - Main fix
- `src/linus/agents/agent/reasoning_agent.py` - Citation extraction from tools
- `src/linus/agents/tools/vector_store.py` - Citation source
- `src/linus/agents/agent/models.py` - Citation data model
- `src/app.py` - FastAPI endpoint citation handling

## Notes

- Citations are aggregated across all subagent executions in all iterations
- Duplicate citations are not removed (by design, to show frequency)
- Citations maintain the order they were collected
- Failed steps return empty citations array
- Skipped steps return empty citations array
