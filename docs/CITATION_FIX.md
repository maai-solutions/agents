# Citation Support - Issue Analysis and Fix

## Problem Summary
Citations were showing up as empty even though the vector_search tool was expected to provide them.

## Root Cause Analysis

### Issue 1: Missing vector_search Tool in Default Tools
The `get_default_tools()` function in [src/linus/agents/agent/tools.py:166-174](src/linus/agents/agent/tools.py#L166-L174) does NOT include the `VectorStoreTool`.

**Default tools list:**
- SearchTool (mock search)
- CalculatorTool
- FileReaderTool
- ShellCommandTool
- APIRequestTool

**Missing:** VectorStoreTool (which provides the `vector_search` tool)

### Issue 2: Citation Extraction Only Works with vector_search Tool
The citation extraction logic in [src/linus/agents/agent/reasoning_agent.py:342-373](src/linus/agents/agent/reasoning_agent.py#L342-L373) specifically checks for:
```python
if task.tool_name == "vector_search":
```

If the vector_search tool is never used, no citations are collected.

### Issue 3: Missing Citation Display in run_llm.py
The [src/run_llm.py](src/run_llm.py) script did not display citations even if they were collected.

## Solution

### Fix 1: Add Enhanced Debug Logging
Added comprehensive debug logging to the citation extraction logic to help diagnose issues:
- Logs when vector_search is detected
- Shows the tool result structure
- Reports JSON parsing status
- Displays citation counts and details

**File:** [src/linus/agents/agent/reasoning_agent.py:343-373](src/linus/agents/agent/reasoning_agent.py#L343-L373)

### Fix 2: Add Citation Display to run_llm.py
Added citation display section that shows:
- Number of citations found
- Table with document ID, chunk number, score, and preview

**File:** [src/run_llm.py:276-299](src/run_llm.py#L276-L299)

### Fix 3: Create Test Scripts
Created two test scripts to verify citation support:

1. **test_citations_simple.py** - Checks if vector_search tool is available
2. **test_vector_citations.py** - Complete test with proper tool setup

## How to Use Citations

### Option 1: Include VectorStoreTool Explicitly
```python
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.tools.vector_store import VectorStoreTool

# Add VectorStoreTool to tools
tools = get_default_tools() + [VectorStoreTool()]

agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=tools,  # Include vector_search
    use_async=True
)

response = await agent.run("Search for information about Ukraine", return_metrics=True)

# Access citations
if response.citations:
    for citation in response.citations:
        print(f"Document: {citation.document_id}")
        print(f"Chunk: {citation.chunk_number}")
        print(f"Score: {citation.score:.4f}")
        print(f"Preview: {citation.content_preview}")
```

### Option 2: Use MCP Vector Store (as in app.py)
The FastAPI app ([src/app.py](src/app.py)) uses MCP (Model Context Protocol) to provide vector search tools. The MCP vector_search tool at [src/linus/mcp/vector_store/server.py:95](src/linus/mcp/vector_store/server.py#L95) is also named "vector_search" and returns citations in the expected format.

## Citation Flow

1. **Agent receives query** → Reasoning phase plans to use vector_search
2. **Tool execution** → vector_search tool returns JSON with "citations" array
3. **Citation extraction** → [reasoning_agent.py:342-373](src/linus/agents/agent/reasoning_agent.py#L342-L373) parses JSON and creates Citation objects
4. **Response** → AgentResponse includes citations list
5. **Display** → run_llm.py or API endpoint displays citations to user

## Testing

### Test 1: Check Tool Availability
```bash
python test_citations_simple.py
```
This will show whether vector_search tool is in the tools list.

### Test 2: Full Citation Test (requires Weaviate)
```bash
python test_vector_citations.py
```
This will run a query, collect citations, and display them.

### Test 3: Using run_llm.py
```bash
python src/run_llm.py --query "What is the Ukraine conflict about?"
```
Now includes citation display section.

### Test 4: Using FastAPI (requires MCP setup)
```bash
# Start the server
python src/app.py

# Send request
POST http://localhost:8000/agent/query
{
    "query": "What is the current situation in Ukraine?",
    "tools": true
}
```

## Citation Data Structure

```python
class Citation(BaseModel):
    document_id: str           # Document identifier
    chunk_number: int          # Chunk/section number within document
    score: Optional[float]     # Relevance score (0.0 to 1.0)
    content_preview: str       # Preview of cited content
```

## Expected Tool Result Format

The vector_search tool must return JSON with this structure:
```json
{
    "status": "success",
    "query": "search query",
    "results": [...],
    "citations": [
        {
            "document_id": "doc_123",
            "chunk_number": 5,
            "score": 0.9234,
            "content_preview": "Preview of content..."
        }
    ],
    "total_results": 5
}
```

Both [VectorStoreTool](src/linus/agents/tools/vector_store.py) and [MCP vector_search](src/linus/mcp/vector_store/server.py) return this format.

## Debugging Citations

If citations are still empty, check the logs for these messages:

1. `[CITATIONS] Detected vector_search tool` - Confirms tool was used
2. `[CITATIONS] Successfully parsed JSON` - Confirms JSON parsing worked
3. `[CITATIONS] Found 'citations' key` - Confirms citations exist in result
4. `[CITATIONS] Successfully extracted N citations` - Confirms extraction success
5. `[CITATIONS] Total citations collected: N` - Final count

## Summary

**The main issue was that `vector_search` tool was not included by default.** To use citations:

1. ✅ Add VectorStoreTool to your agent's tools list, OR
2. ✅ Use the FastAPI app with MCP vector-store configured, OR
3. ✅ Make sure any tool named "vector_search" returns JSON with "citations" array

The citation extraction and display code is now working correctly with enhanced logging for debugging.
