# MCP Vector Store Server

A Model Context Protocol (MCP) server that provides vector store search capabilities using Weaviate.

## Overview

This MCP server exposes vector store operations through the MCP protocol, allowing AI assistants and other MCP clients to search document content using hybrid search (combining vector and keyword search).

## Features

- **Hybrid Search**: Combines vector similarity search with keyword matching
- **Configurable Parameters**: Adjust search parameters like alpha (vector/keyword balance), limit, and distance threshold
- **Docker Support**: Easy deployment with Docker and docker-compose
- **Weaviate Integration**: Uses Weaviate as the vector database
- **OpenAI-Compatible Embeddings**: Works with any OpenAI-compatible API (Ollama, OpenAI, etc.)

## Architecture

```
┌─────────────┐         ┌──────────────────┐         ┌──────────┐
│ MCP Client  │ ◄─────► │  MCP Server      │ ◄─────► │ Weaviate │
│ (AI Agent)  │  stdio  │ (vector_store)   │  HTTP   │ Database │
└─────────────┘         └──────────────────┘         └──────────┘
                               │
                               │ HTTP
                               ▼
                        ┌──────────┐
                        │  Ollama  │
                        │ (Embed)  │
                        └──────────┘
```

## Available Tools

### `vector_search`

Search for information in document content and text chunks.

**Parameters:**
- `query` (required): Natural language question or topic to search for
- `collection` (optional): Weaviate collection name (default: from settings)
- `max_distance` (optional): Maximum distance for results (default: 0.7)
- `alpha` (optional): Hybrid search parameter, 0=keyword, 1=vector (default: 0.5)
- `limit` (optional): Maximum number of results (default: 5)

**Returns:**
Formatted string with search results including:
- Score for each result
- Content preview (first 500 chars)
- Metadata

## Installation

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables (see Configuration section)

3. Run the server:
```bash
python -m linus.mcp.vector_store
```

### Docker Deployment

#### Option 1: Docker Compose (Recommended)

This will start the MCP server along with Weaviate and Ollama:

```bash
cd src/linus/mcp/vector_store
docker-compose up -d
```

To view logs:
```bash
docker-compose logs -f mcp-vector-store
```

To stop:
```bash
docker-compose down
```

#### Option 2: Docker Build Only

Build the Docker image:
```bash
cd /Users/udg/Projects/Git/agents
docker build -f src/linus/mcp/vector_store/Dockerfile -t mcp-vector-store:latest .
```

Run the container:
```bash
docker run -it \
  -e WV_HTTP_HOST=weaviate \
  -e WV_HTTP_PORT=8080 \
  -e LLM_API_BASE=http://ollama:11434/v1 \
  --network mcp-network \
  mcp-vector-store:latest
```

## Configuration

### Environment Variables

#### Weaviate Configuration
- `WV_HTTP_HOST`: Weaviate HTTP host (default: `localhost`)
- `WV_HTTP_PORT`: Weaviate HTTP port (default: `18080`)
- `WV_HTTP_SCHEME`: HTTP scheme - `http` or `https` (default: `http`)
- `WV_GRPC_HOST`: Weaviate gRPC host (default: `localhost`)
- `WV_GRPC_PORT`: Weaviate gRPC port (default: `50051`)
- `WV_GRPC_SCHEME`: gRPC scheme (default: `http`)
- `WV_COLLECTION`: Default collection name (default: `arag_dev`)
- `WV_MAX_DISTANCE`: Maximum distance threshold (default: `0.7`)
- `WV_ALPHA`: Hybrid search alpha parameter (default: `0.5`)
- `WV_LIMIT`: Default result limit (default: `5`)
- `WV_EMBEDDING_MODEL`: Embedding model name (default: `nomic-embed-text:latest`)

#### LLM Configuration (for embeddings)
- `LLM_API_BASE`: OpenAI-compatible API endpoint (default: `http://localhost:11434/v1`)
- `LLM_MODEL`: Model name (default: `gemma3:27b`)
- `LLM_API_KEY`: API key (default: `not-needed` for Ollama)

#### Telemetry (optional)
- `TELEMETRY_ENABLED`: Enable/disable telemetry (default: `false`)

### Configuration Files

Create a `.env` file in the project root or use environment variables:

```bash
# Weaviate
WV_HTTP_HOST=localhost
WV_HTTP_PORT=18080
WV_COLLECTION=arag_dev

# LLM (for embeddings)
LLM_API_BASE=http://localhost:11434/v1
LLM_API_KEY=not-needed
WV_EMBEDDING_MODEL=nomic-embed-text:latest
```

## Usage Example

### Using with MCP Client

```json
{
  "tool": "vector_search",
  "arguments": {
    "query": "What are the benefits of vector databases?",
    "limit": 3,
    "alpha": 0.7
  }
}
```

### Response Format

```
Content search results for 'What are the benefits of vector databases?' (alpha=0.7, limit=3, max_distance=0.7):

1. [Score: 0.8543]
   Content: Vector databases offer several advantages for AI applications. They enable semantic search by understanding the meaning...
   Metadata: {'source': 'docs/vectordb.md', 'chunk_id': 'abc123'}

2. [Score: 0.7821]
   Content: The main benefits include fast similarity search, scalability for large datasets, and native support for embeddings...
   Metadata: {'source': 'blog/2024-01-15.md', 'author': 'Jane Doe'}
```

## MCP Protocol

This server implements the Model Context Protocol (MCP) specification:
- Uses stdio for communication
- Supports `list_tools` to discover available tools
- Supports `call_tool` to execute searches
- Returns results as TextContent

## Troubleshooting

### Connection Issues

1. **Weaviate not accessible:**
   - Verify Weaviate is running: `curl http://localhost:18080/v1/.well-known/ready`
   - Check network connectivity in Docker: `docker network inspect mcp-network`

2. **Ollama not accessible:**
   - Verify Ollama is running: `curl http://localhost:11434/api/tags`
   - Ensure the embedding model is pulled: `docker exec ollama ollama pull nomic-embed-text:latest`

3. **Collection not found:**
   - Verify collection exists in Weaviate
   - Check `WV_COLLECTION` environment variable

### Performance

- Adjust `WV_LIMIT` to control result count
- Tune `WV_ALPHA` for better keyword vs vector balance (0.0 = keyword only, 1.0 = vector only)
- Increase `WV_MAX_DISTANCE` to get more results

## Development

### Project Structure

```
src/linus/mcp/
├── __init__.py                    # MCP module root
└── vector_store/                  # Vector Store MCP Server
    ├── __init__.py
    ├── __main__.py                # Module entry point
    ├── server.py                  # Main MCP server implementation
    ├── test_server.py             # Test script
    ├── Dockerfile                 # Docker container definition
    ├── docker-compose.yml         # Multi-container orchestration
    ├── requirements.txt           # Python dependencies
    ├── .env.example               # Configuration template
    ├── Makefile                   # Development commands
    ├── README.md                  # This file
    └── QUICKSTART.md             # Quick start guide
```

### Testing

Test the server locally:

```bash
# Start dependencies
docker-compose up -d weaviate ollama

# Run the MCP server
python -m linus.mcp.vector_store
```

### Adding New Tools

To add new tools to the MCP server:

1. Add a new tool definition in `list_tools()`
2. Add handler logic in `call_tool()`
3. Implement the tool method in the `VectorStoreMCPServer` class

## License

See the main project LICENSE file.
