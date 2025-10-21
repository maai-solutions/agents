# Citation Support

The agent framework now includes built-in support for citations, allowing you to track the sources of information returned by the vector search tool.

## Overview

When the `vector_search` tool returns results from the Weaviate vector store, it now includes citation information (document ID and chunk number) for each result. The agent automatically extracts these citations and includes them in the response.

## Features

- **Automatic Citation Extraction**: Citations are automatically extracted from `vector_search` tool results
- **Structured Citation Format**: Each citation includes document ID, chunk number, relevance score, and content preview
- **API Integration**: Citations are included in all API responses
- **Pydantic Models**: Type-safe citation handling throughout the system

## Citation Model

```python
@dataclass
class Citation:
    """Citation for a piece of information from a document."""
    document_id: str          # Unique identifier for the source document
    chunk_number: int         # Chunk/section number within the document
    score: Optional[float]    # Relevance score (0.0 to 1.0)
    content_preview: Optional[str]  # Preview of the cited content
```

## Vector Store Tool Response Format

The `vector_search` tool now returns structured JSON with citations:

```json
{
  "status": "success",
  "query": "What is the relationship between Russia and Ukraine?",
  "search_params": {
    "alpha": 0.75,
    "limit": 5,
    "max_distance": 0.7
  },
  "results": [
    {
      "rank": 1,
      "score": 0.9234,
      "content": "Full text content...",
      "document_id": "doc_12345",
      "chunk_number": 3,
      "metadata": {...}
    }
  ],
  "citations": [
    {
      "document_id": "doc_12345",
      "chunk_number": 3,
      "score": 0.9234,
      "content_preview": "Preview text..."
    }
  ],
  "total_results": 1
}
```

## Using Citations in Agent Code

### Basic Agent Usage

```python
from linus.agents.agent.factory import Agent
from linus.agents.tools.vector_store import VectorStoreTool

# Create agent with vector search tool
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=[VectorStoreTool()],
    use_async=True
)

# Run query
response = await agent.run("What is the Ukraine conflict about?")

# Access citations
if response.citations:
    for citation in response.citations:
        print(f"Source: {citation.document_id}, Chunk: {citation.chunk_number}")
        print(f"Score: {citation.score}")
        print(f"Preview: {citation.content_preview}")
```

### AgentResponse with Citations

```python
from linus.agents.agent.models import AgentResponse

# Run agent with metrics
response = await agent.run(query, return_metrics=True)

# Response is AgentResponse object with citations
print(f"Result: {response.result}")
print(f"Citations: {len(response.citations)}")

# Convert to dictionary for API responses
response_dict = response.to_dict()
# response_dict['citations'] contains list of citation dictionaries
```

## API Response Format

### Query Endpoint

```http
POST /agent/query
Content-Type: application/json

{
  "query": "What is the relationship between Russia and Ukraine?"
}
```

**Response:**

```json
{
  "query": "What is the relationship between Russia and Ukraine?",
  "response": "Based on the search results...",
  "citations": [
    {
      "document_id": "doc_12345",
      "chunk_number": 3,
      "score": 0.9234,
      "content_preview": "Russia and Ukraine have had a complex historical relationship..."
    }
  ],
  "subagents_used": ["research_agent"],
  "execution_time": 3.45,
  "timestamp": "2025-01-15T10:30:00",
  "metrics": {...}
}
```

## Citation Flow

### With ReasoningAgent (Direct)

1. **User Query**: User asks a question that requires searching documents
2. **Agent Reasoning**: Agent determines it needs to use `vector_search` tool
3. **Vector Search**: Tool queries Weaviate and returns structured JSON with citations
4. **Citation Extraction**: Agent automatically extracts citation data from tool result
5. **Response Assembly**: Agent includes citations in AgentResponse
6. **API Response**: FastAPI endpoint returns citations to client

### With CoordinatorAgent (Multi-Agent)

1. **User Query**: User asks a question
2. **Coordinator Planning**: Creates plan with subagent steps (e.g., Researcher)
3. **Subagent Execution**: Each subagent runs independently
   - Subagent uses `vector_search` tool
   - Tool returns JSON with citations
   - Subagent extracts citations into its AgentResponse
4. **Citation Collection**: Coordinator collects citations from each subagent
5. **Citation Aggregation**: All citations from all steps are combined
6. **Response Assembly**: Coordinator includes all citations in final AgentResponse
7. **API Response**: FastAPI endpoint returns aggregated citations to client

## Weaviate Schema Requirements

For citations to work properly, your Weaviate collection should include these properties:

- `document_id` or `doc_id`: Unique identifier for the source document
- `chunk_number` or `chunk_id`: Chunk/section number within the document
- `text` or `content`: The actual text content
- Other metadata fields (optional)

Example Weaviate object:

```python
{
    "text": "Full text content of the chunk...",
    "document_id": "article_2024_01_15",
    "chunk_number": 5,
    "title": "Article Title",
    "author": "Author Name",
    "date": "2024-01-15"
}
```

## Implementation Details

### ReasoningAgent Citation Collection

The `ReasoningAgent` automatically collects citations during task execution:

```python
# In ReasoningAgent._run_with_trace()
citations = []  # Initialize citation collection

# During tool execution
if task.tool_name == "vector_search":
    try:
        result_data = json.loads(task_result)
        if "citations" in result_data:
            for citation_data in result_data["citations"]:
                citation = Citation(
                    document_id=citation_data.get("document_id", "unknown"),
                    chunk_number=citation_data.get("chunk_number", 0),
                    score=citation_data.get("score"),
                    content_preview=citation_data.get("content_preview")
                )
                citations.append(citation)
    except (json.JSONDecodeError, KeyError):
        # Tool didn't return valid citation data
        pass

# Return citations in response
return AgentResponse(
    result=formatted_result,
    metrics=metrics,
    execution_history=execution_history,
    citations=citations
)
```

### FastAPI Integration

The FastAPI endpoint extracts citations from the agent response:

```python
# Extract citations from AgentResponse
citations = []
if agent_response.citations:
    citations = [
        Citation(
            document_id=citation.document_id,
            chunk_number=citation.chunk_number,
            score=citation.score,
            content_preview=citation.content_preview
        )
        for citation in agent_response.citations
    ]

# Include in API response
response = AgentResponse(
    query=request.query,
    response=result_text,
    citations=citations,
    ...
)
```

## Testing

Run the citation test script:

```bash
python test_citations.py
```

This will:
1. Run an agent query using vector search
2. Display the response with citations
3. Show the citation format in API responses
4. Display metrics and execution history

## Best Practices

1. **Unique Document IDs**: Ensure each document in Weaviate has a unique `document_id`
2. **Sequential Chunk Numbers**: Number chunks sequentially within each document
3. **Meaningful Previews**: Keep content previews concise (100-200 characters)
4. **Score Interpretation**: Higher scores (closer to 1.0) indicate more relevant results
5. **Citation Display**: Show citations to users for transparency and verification

## Future Enhancements

Potential improvements for citation support:

- **Citation Formatting**: Add citation formatter for different styles (APA, MLA, etc.)
- **Deduplication**: Remove duplicate citations from the same document/chunk
- **Inline Citations**: Embed citation references directly in the response text
- **Citation Linking**: Include URLs or paths to original documents
- **Multi-Tool Citations**: Support citations from other tools beyond vector_search
- **Citation Ranking**: Sort citations by relevance score or document importance

## Troubleshooting

### No Citations Returned

If citations are not appearing in responses:

1. **Check Tool Results**: Verify that `vector_search` is being used and returning results
2. **Verify Weaviate Schema**: Ensure documents have `document_id` and `chunk_number` fields
3. **Enable Verbose Logging**: Set `verbose=True` on agent to see citation extraction logs
4. **Check JSON Format**: Verify tool returns valid JSON with `citations` array

### Missing Citation Fields

If some citation fields are missing:

- `document_id` defaults to `"unknown"` if not found
- `chunk_number` defaults to `0` if not found
- `score` and `content_preview` are optional and may be `None`

### Example Debug Log

```
[CITATIONS] Extracted 3 citations from vector_search
[CITATIONS] Could not extract citations from tool result: Expecting value: line 1 column 1 (char 0)
```

The second message indicates the tool result was not valid JSON, which may occur if the tool encountered an error.

## Related Documentation

- [Vector Store Tool](./VECTOR_STORE.md) - Details on the vector search tool
- [Agent Architecture](./ARCHITECTURE.md) - Overall agent system design
- [API Documentation](./API.md) - Complete API reference
- [Weaviate Integration](./WEAVIATE.md) - Weaviate setup and configuration
