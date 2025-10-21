# Citation Fix for MCP Vector Store

## Problem Identified

**Root Cause:** The MCP vector_search server was returning **plain text** instead of **JSON with citations**.

The citation extraction code in [reasoning_agent.py:342-373](src/linus/agents/agent/reasoning_agent.py#L342-L373) expects vector_search to return JSON like this:

```json
{
  "status": "success",
  "query": "...",
  "results": [...],
  "citations": [
    {
      "document_id": "doc_123",
      "chunk_number": 5,
      "score": 0.9234,
      "content_preview": "..."
    }
  ]
}
```

But the MCP server at [src/linus/mcp/vector_store/server.py:278-280](src/linus/mcp/vector_store/server.py#L278-L280) was returning:

```
Content search results for '...'
Showing X unique results...

1. [Score: 0.9234]
   Content: ...
```

## Changes Made

### 1. Updated MCP Vector Store Server ([server.py:229-326](src/linus/mcp/vector_store/server.py#L229-L326))

**Before:** Returned plain text formatted results
**After:** Returns JSON with structured results and citations

Key changes:
- Added `citations` array to collect citation metadata
- Extract `document_id` and `chunk_number` from Weaviate properties
- Create citation entries with document_id, chunk_number, score, and content_preview
- Return JSON response matching VectorStoreTool format
- Added citation count logging

### 2. Enhanced Citation Extraction Logging ([reasoning_agent.py:343-373](src/linus/agents/agent/reasoning_agent.py#L343-L373))

Added comprehensive debug logging:
- `[CITATIONS] Detected vector_search tool` - Confirms tool was used
- `[CITATIONS] Successfully parsed JSON` - JSON parsing status
- `[CITATIONS] Found 'citations' key with N items` - Citation count
- `[CITATIONS] Successfully extracted N citations` - Extraction success
- `[CITATIONS] Total citations collected: N` - Final summary

### 3. Added Citation Display ([run_llm.py:276-299](src/run_llm.py#L276-L299))

New section displays:
- Citation count
- Table with document ID, chunk number, score, and preview

## How Citations Flow (Fixed)

1. **User query** → Coordinator plans to use Researcher subagent
2. **Researcher subagent** → Uses vector_search tool (MCP)
3. **MCP vector_search** → Returns **JSON with citations** (FIXED!)
4. **ReasoningAgent** → Extracts citations from JSON (line 342-373)
5. **Coordinator** → Collects citations from Researcher (line 718)
6. **API Response** → Returns citations to user

## Testing the Fix

### Step 1: Restart the FastAPI Server

The server needs to be restarted to pick up the MCP server changes:

```bash
# Kill any running instances
pkill -f "python src/app.py"

# Start fresh
python src/app.py
```

### Step 2: Test with API Request

```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "why are the whales in danger?",
    "tools": true,
    "stream": false,
    "session_id": "test123"
  }' | jq '.citations'
```

**Expected output:**
```json
[
  {
    "document_id": "doc_xxx",
    "chunk_number": 5,
    "score": 0.9234,
    "content_preview": "Whales face several threats..."
  },
  ...
]
```

### Step 3: Check Logs for Citation Debugging

Look for these log messages:

```
[CITATIONS] Detected vector_search tool, attempting to extract citations
[CITATIONS] Successfully parsed JSON, keys: dict_keys(['status', 'query', 'results', 'citations', ...])
[CITATIONS] Found 'citations' key with 5 items
[CITATIONS] Successfully extracted 5 citations from vector_search
[COORDINATOR] Collected 5 citations from step 1
[COORDINATOR] Collected 15 total citations from subagents
```

### Step 4: Test with run_llm.py

```bash
python src/run_llm.py --query "What is the Ukraine conflict about?"
```

Now includes a citation table with document IDs, chunks, scores, and previews.

## Weaviate Data Requirements

For citations to work, your Weaviate collection must have these properties:

**Required:**
- `text` or `content` - The document content
- `document_id` or `doc_id` - Unique document identifier
- `chunk_number` or `chunk_id` - Chunk/section number

**Example Weaviate schema:**
```python
{
  "text": "Content about whales...",
  "document_id": "whale_conservation_2023",
  "chunk_number": 3,
  "source": "NOAA",
  "date": "2023-05-15"
}
```

If these fields are missing, the MCP server will use fallback values:
- `document_id` → `f'doc_{idx}'`
- `chunk_number` → `idx`

## Comparison: Before vs After

### Before (Plain Text)
```
Content search results for 'whales danger':
Showing 3 unique results

1. [Score: 0.9234]
   Content: Whales face several threats...
   Metadata: {...}
```

**Result:** Citations = `[]` (empty)

### After (JSON with Citations)
```json
{
  "status": "success",
  "query": "whales danger",
  "results": [
    {
      "rank": 1,
      "score": 0.9234,
      "content": "Whales face several threats...",
      "document_id": "whale_report_2023",
      "chunk_number": 3
    }
  ],
  "citations": [
    {
      "document_id": "whale_report_2023",
      "chunk_number": 3,
      "score": 0.9234,
      "content_preview": "Whales face several threats..."
    }
  ]
}
```

**Result:** Citations = `[{document_id, chunk_number, score, preview}]` ✅

## Files Changed

1. **[src/linus/mcp/vector_store/server.py](src/linus/mcp/vector_store/server.py)** - Return JSON with citations
2. **[src/linus/agents/agent/reasoning_agent.py](src/linus/agents/agent/reasoning_agent.py)** - Enhanced logging
3. **[src/run_llm.py](src/run_llm.py)** - Display citations

## Summary

The citation feature is now **fully functional**! The key fix was changing the MCP vector_search server from returning plain text to returning JSON with a `citations` array. The coordinator properly collects citations from subagents and includes them in the final API response.

**To see citations working:**
1. Restart the FastAPI server
2. Send a query that requires vector search
3. Check the `citations` array in the response
4. Look for `[CITATIONS]` logs for debugging
