# Quick Start Guide - MCP Vector Store Server

Get the MCP Vector Store Server running in 5 minutes!

## Prerequisites

- Docker and Docker Compose installed
- Git repository cloned
- 8GB+ RAM recommended (for Ollama and Weaviate)

## Option 1: Docker Compose (Easiest)

### 1. Navigate to Vector Store MCP directory

```bash
cd /Users/udg/Projects/Git/agents/src/linus/mcp/vector_store
```

### 2. Start all services

```bash
make up
# or
docker-compose up -d
```

This will start:
- MCP Vector Store Server
- Weaviate (vector database)
- Ollama (for embeddings)

### 3. Pull required models

```bash
make pull-models
# or
docker exec ollama ollama pull nomic-embed-text:latest
docker exec ollama ollama pull gemma3:27b
```

### 4. Check services are running

```bash
make ps
# or
docker-compose ps
```

### 5. View logs

```bash
make logs-mcp
# or
docker-compose logs -f mcp-vector-store
```

### 6. Test the server

```bash
# First, populate Weaviate with some test data (see DATA_SETUP.md)
# Then run a test search
docker-compose exec mcp-vector-store python -m linus.mcp.vector_store.test_server
```

## Option 2: Local Development

### 1. Install dependencies

```bash
cd /Users/udg/Projects/Git/agents
pip install -r src/linus/mcp/vector_store/requirements.txt
```

### 2. Start Weaviate and Ollama

```bash
cd src/linus/mcp/vector_store
docker-compose up -d weaviate ollama
```

### 3. Configure environment

Copy and edit `.env` file:

```bash
cp src/linus/mcp/vector_store/.env.example .env
```

Edit `.env` to set your configuration.

### 4. Run the MCP server

```bash
cd /Users/udg/Projects/Git/agents
python -m linus.mcp.vector_store
```

## Using the MCP Server

### With MCP-Compatible Clients

Configure your MCP client (like Claude Desktop) to use the server:

```json
{
  "mcpServers": {
    "vector-store": {
      "command": "python",
      "args": ["-m", "linus.mcp.vector_store"],
      "cwd": "/Users/udg/Projects/Git/agents/src",
      "env": {
        "WV_HTTP_HOST": "localhost",
        "WV_HTTP_PORT": "18080",
        "LLM_API_BASE": "http://localhost:11434/v1"
      }
    }
  }
}
```

### Example Query

Once connected, you can query the vector store:

```
Search for information about machine learning algorithms
```

The MCP server will use the `vector_search` tool to find relevant content.

## Common Commands

### Start services
```bash
make up
```

### Stop services
```bash
make down
```

### View logs
```bash
make logs
```

### Restart services
```bash
make restart
```

### Clean up everything
```bash
make clean
```

### Check service health
```bash
make check-weaviate
make check-ollama
```

## Troubleshooting

### Services won't start

1. Check ports are available:
```bash
lsof -i :18080  # Weaviate HTTP
lsof -i :50051  # Weaviate gRPC
lsof -i :11434  # Ollama
```

2. Check Docker resources:
```bash
docker system df
```

### Weaviate connection error

```bash
# Check Weaviate is running
curl http://localhost:18080/v1/.well-known/ready

# Check collections
curl http://localhost:18080/v1/schema
```

### Ollama model not found

```bash
# List available models
docker exec ollama ollama list

# Pull missing model
docker exec ollama ollama pull nomic-embed-text:latest
```

### Collection not found

Create a collection in Weaviate first. Example:

```python
import weaviate

client = weaviate.connect_to_local()
collection = client.collections.create(
    name="arag_dev",
    # Add your schema here
)
```

## Next Steps

1. **Populate data**: Add documents to your Weaviate collection
2. **Configure parameters**: Adjust `WV_ALPHA`, `WV_LIMIT`, etc. in `.env`
3. **Test queries**: Use the test script or MCP client to search
4. **Monitor**: Check logs and telemetry (if enabled)

## Resources

- [README.md](README.md) - Full documentation
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Weaviate Docs](https://weaviate.io/developers/weaviate)
- [Ollama Docs](https://ollama.ai/)

## Getting Help

If you encounter issues:

1. Check logs: `make logs`
2. Verify configuration in `.env`
3. Test individual components (Weaviate, Ollama)
4. Review [README.md](README.md) for detailed troubleshooting
