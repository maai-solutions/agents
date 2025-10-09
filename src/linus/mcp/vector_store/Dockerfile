# Multi-stage build for MCP Vector Store Server
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY .env.example .env

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose port (if needed for debugging/health checks)
# MCP servers typically use stdio, but we expose a port for potential HTTP endpoints
EXPOSE 8000

# Set environment variables with defaults
ENV WV_HTTP_HOST=localhost \
    WV_HTTP_PORT=18080 \
    WV_HTTP_SCHEME=http \
    WV_GRPC_HOST=localhost \
    WV_GRPC_PORT=50051 \
    WV_GRPC_SCHEME=http \
    WV_COLLECTION=arag_dev \
    WV_MAX_DISTANCE=0.7 \
    WV_ALPHA=0.5 \
    WV_LIMIT=5 \
    WV_EMBEDDING_MODEL=nomic-embed-text:latest \
    LLM_API_BASE=http://localhost:11434/v1 \
    LLM_MODEL=gemma3:27b \
    LLM_API_KEY=not-needed

# Run the MCP server
CMD ["python", "-m", "linus.mcp.vector_store"]
