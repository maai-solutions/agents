# Citation System Improvements

## Summary

The citation system has been improved to be more abstract and flexible, with inline citations and a proper references section.

## Changes Made

### 1. Tool-Agnostic Citation Extraction

**Before:** Citations were only extracted from the `vector_search` tool specifically.

**After:** Citations are now extracted from ANY tool that returns structured JSON data containing a `citations` field.

**Location:** [reasoning_agent.py:341-371](src/linus/agents/agent/reasoning_agent.py#L341-L371)

```python
# Extract citations from any tool that returns structured data with citations
# Tool-agnostic: check if the result contains a "citations" field
try:
    result_data = json.loads(task_result)
    if "citations" in result_data:
        citation_list = result_data["citations"]
        # Extract citations...
except json.JSONDecodeError:
    # Tool result is not JSON, skip
    pass
```

### 2. Inline Citations in Response

**Feature:** The LLM now includes inline citation markers `[1]`, `[2]`, etc. directly in the response text, next to the sentences/paragraphs they support.

**Example:**
```
Ukraine is a country located in Eastern Europe, sharing a border with Russia
to the east and northeast [1, 4, 7]. The conflict began in 2014 when Russia
annexed Crimea [2, 5, 8].
```

### 3. References Section at Bottom

**Feature:** A "References" section is automatically added at the bottom of responses, listing all citations with their `document_id` and `chunk_number`.

**Example:**
```
## References
[1] Document: doc_ukraine_001, Chunk: 1
[2] Document: doc_conflict_002, Chunk: 3
[3] Document: doc_conflict_002, Chunk: 5
```

**Location:** [reasoning_agent.py:624-653](src/linus/agents/agent/reasoning_agent.py#L624-L653)

### 4. Enhanced Prompt Instructions

The final response formatting prompt now includes specific instructions for:
- Including inline citations using square brackets
- Placing citation numbers after supported claims
- Creating a References section with proper formatting

## Tool Requirements

For a tool to support citations, it must return JSON with a `citations` array:

```json
{
  "status": "success",
  "results": [...],
  "citations": [
    {
      "document_id": "doc_123",
      "chunk_number": 1,
      "score": 0.95,
      "content_preview": "Preview text..."
    }
  ]
}
```

### Citation Object Structure

Each citation must contain:
- `document_id` (required): Unique identifier for the source document
- `chunk_number` (required): Chunk/section number within the document
- `score` (optional): Relevance score (0.0 to 1.0)
- `content_preview` (optional): Preview of the cited content

## Existing Tools with Citation Support

### VectorStoreTool
- **Location:** [src/linus/agents/tools/vector_store.py](src/linus/agents/tools/vector_store.py)
- **Returns:** Structured JSON with `citations` array containing document_id, chunk_number, score, and content_preview
- **Already compatible:** Yes, no changes needed

### MCP Vector Store Server
- **Location:** [src/linus/mcp/vector_store/server.py](src/linus/mcp/vector_store/server.py)
- **Returns:** Same format as VectorStoreTool
- **Already compatible:** Yes, no changes needed

## Testing

A test script is provided to verify citation functionality:

```bash
python test_citation_mock.py
```

This test uses a mock tool that returns citations and verifies:
1. Citations are extracted from tool results
2. Inline citation markers appear in the response
3. References section is populated with citation details

## Example Usage

```python
from linus.agents.agent.factory import Agent
from linus.agents.tools.vector_store import VectorStoreTool
from linus.agents.agent.tools import get_default_tools

# Create agent with vector search tool
tools = get_default_tools() + [VectorStoreTool()]
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=tools,
    use_async=True
)

# Run query
response = await agent.run("What is the Ukraine conflict about?", return_metrics=True)

# Access result with inline citations
print(response.result)
# Output: "Ukraine is a country in Eastern Europe [1]. The conflict began in
#          2014 when Russia annexed Crimea [2]...
#
#          ## References
#          [1] Document: doc_ukraine_001, Chunk: 1
#          [2] Document: doc_conflict_002, Chunk: 3"

# Access extracted citations programmatically
for citation in response.citations:
    print(f"Source: {citation.document_id}, Chunk: {citation.chunk_number}")
```

## Benefits

1. **Transparency:** Users can see exactly which sources support each claim
2. **Verifiability:** Citations include document_id and chunk_number for easy lookup
3. **Extensibility:** Any tool can add citation support by including a `citations` field
4. **Backward Compatible:** Tools without citations continue to work normally

## Implementation Details

- Citations are collected during the execution phase from all tool results
- The `_format_final_response_with_history()` method receives the citations list
- Citation information is added to the LLM prompt along with the research findings
- The LLM is instructed to include inline citations and create a References section
- The system prompt dynamically changes based on whether citations are available

## Future Enhancements

Possible improvements for the future:
1. Deduplicate citations (currently same citation can appear multiple times)
2. Add citation hover tooltips in rich terminal output
3. Support for different citation formats (e.g., APA, MLA)
4. Citation validation (warn if cited but source not in citations list)
5. Smart citation numbering (reuse numbers for repeated sources)
