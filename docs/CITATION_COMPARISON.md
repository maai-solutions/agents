# Citation System: Before vs After

## Before

### Citation Extraction
- **Hardcoded:** Only worked with `vector_search` tool specifically
- **Location:** Line 342 in reasoning_agent.py
```python
if task.tool_name == "vector_search":
    # Extract citations...
```

### Response Format
```
Ukraine is a country in Eastern Europe. It shares borders with Russia.
The conflict began in 2014 when Russia annexed Crimea from Ukraine.
The situation escalated in February 2022 with a full-scale invasion.
```

**Issues:**
- No way to know which source supports which claim
- Citations were extracted but not integrated into the response
- No references section
- Users had to manually check `response.citations` array

## After

### Citation Extraction
- **Tool-agnostic:** Works with ANY tool that returns `citations` field
- **Location:** Lines 341-371 in reasoning_agent.py
```python
# Extract citations from any tool that returns structured data with citations
try:
    result_data = json.loads(task_result)
    if "citations" in result_data:
        citation_list = result_data["citations"]
        # Extract citations from ANY tool
except json.JSONDecodeError:
    pass  # Tool doesn't return JSON, skip
```

### Response Format
```
Ukraine is a country located in Eastern Europe, sharing a border with Russia
to the east and northeast [1, 4, 7]. The conflict began in 2014 when Russia
annexed Crimea [2, 5, 8]. The situation dramatically escalated in February
2022 with a full-scale invasion by Russian forces [3, 6, 9].

## References
[1] Document: doc_ukraine_001, Chunk: 1
[2] Document: doc_conflict_002, Chunk: 3
[3] Document: doc_conflict_002, Chunk: 5
[4] Document: doc_ukraine_001, Chunk: 1
[5] Document: doc_conflict_002, Chunk: 3
[6] Document: doc_conflict_002, Chunk: 5
[7] Document: doc_ukraine_001, Chunk: 1
[8] Document: doc_conflict_002, Chunk: 3
[9] Document: doc_conflict_002, Chunk: 5
```

**Benefits:**
- ✅ Clear inline citations showing which sources support each claim
- ✅ Complete references section at the bottom
- ✅ Easy to verify claims by looking up document_id and chunk_number
- ✅ Professional academic-style formatting

## Implementation Comparison

### Old Prompt (No Citation Instructions)
```python
system_prompt = """You are an assistant that synthesizes information...
1. Directly answers the original question
2. Integrates information from all findings
3. Provides clear, factual information
4. Is well-organized and easy to read"""
```

### New Prompt (With Citation Instructions)
```python
system_prompt = """You are an assistant that synthesizes information...
1. Directly answers the original question
2. Integrates information from all findings
3. Provides clear, factual information
4. Is well-organized and easy to read"""

if citations:
    system_prompt += """
5. CRITICAL: Include inline citations using [1], [2], etc.
6. Place citation numbers after the sentence or claim they support
7. At the end, include a References section listing ALL citations:

## References
[1] Document: document_id, Chunk: chunk_number
[2] Document: document_id, Chunk: chunk_number"""
```

## Testing Results

### Mock Tool Test Output

```bash
$ python test_citation_mock.py

================================================================================
FINAL RESPONSE
================================================================================

Ukraine is a country located in Eastern Europe, sharing a border with Russia
to the east and northeast [1, 4, 7]. The current conflict involving Ukraine
has a history dating back to 2014, when Russia annexed Crimea [2, 5, 8].
The situation dramatically escalated further in February 2022 with a
full-scale invasion of Ukraine by Russian forces [3, 6, 9].

## References
[1] Document: doc_ukraine_001, Chunk: 1
[2] Document: doc_conflict_002, Chunk: 3
[3] Document: doc_conflict_002, Chunk: 5
[4] Document: doc_ukraine_001, Chunk: 1
[5] Document: doc_conflict_002, Chunk: 3
[6] Document: doc_conflict_002, Chunk: 5
[7] Document: doc_ukraine_001, Chunk: 1
[8] Document: doc_conflict_002, Chunk: 3
[9] Document: doc_conflict_002, Chunk: 5

================================================================================
INLINE CITATION ANALYSIS
================================================================================
✅ Found inline citation markers: [1, 2, 3, 4, 5, 6, 7, 8, 9]
✅ Found 'References' section in response
```

## Key Improvements Summary

| Feature | Before | After |
|---------|--------|-------|
| Citation extraction | `vector_search` only | Any tool with `citations` field |
| Inline citations | ❌ No | ✅ Yes [1], [2], etc. |
| References section | ❌ No | ✅ Yes, at bottom |
| Verifiability | ❌ Hard | ✅ Easy (doc_id + chunk) |
| Tool compatibility | 1 tool | Unlimited tools |
| User experience | Manual lookup | Integrated in response |

## Migration Guide for Custom Tools

If you have a custom tool that should support citations, simply return JSON with a `citations` array:

```python
class MyCustomTool(BaseTool):
    def _run(self, query: str) -> str:
        # Your tool logic...
        
        result = {
            "status": "success",
            "data": your_data,
            "citations": [  # Add this!
                {
                    "document_id": "my_doc_123",
                    "chunk_number": 5,
                    "score": 0.95,
                    "content_preview": "Preview text..."
                }
            ]
        }
        return json.dumps(result)
```

That's it! The agent will automatically:
1. Extract the citations
2. Include inline references in the response
3. Add a References section at the bottom
