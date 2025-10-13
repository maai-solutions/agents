"""FastAPI application with CoordinatorAgent orchestrating specialized subagents."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import json
import asyncio
import os
import sys
from datetime import datetime
from loguru import logger

from dotenv import load_dotenv
import os
from pathlib import Path

from linus.agents.agent.mcp_client import MCPServerConfig, connect_mcp_servers
from linus.agents.tools.entities_search import EntitiesSearchTool

# Load .env from src directory (where this file is located)
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from linus.agents.agent import Agent, CoordinatorAgent, SubAgent
from linus.agents.telemetry import initialize_telemetry
from linus.settings import Settings


# Load settings first
settings = Settings()

# Configure logging early with rich support using settings
if settings.log_file:
    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)

# Rich console handler for beautiful terminal output
from rich.console import Console
from rich.logging import RichHandler

console = Console()

# Remove default handler and add rich console handler
logger.remove()
logger.add(
    RichHandler(
        console=console,
        rich_tracebacks=settings.log_rich_tracebacks,
        tracebacks_show_locals=True,
        markup=True,
        show_time=settings.log_show_time,
        show_level=True,
        show_path=settings.log_show_path
    ),
    format="{message}",
    level=settings.log_console_level
)

# File handler for detailed logs (if configured)
if settings.log_file:
    logger.add(
        settings.log_file,
        rotation=settings.log_file_rotation,
        retention=settings.log_file_retention,
        compression=settings.log_file_compression,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level=settings.log_file_level
    )

# Global agent instance
coordinator: Optional[CoordinatorAgent] = None
mcp_manager = None

# Store conversation history
conversation_history: List[Dict[str, Any]] = []


class AgentRequest(BaseModel):
    """Request model for agent interactions."""

    query: str = Field(..., description="The user's query or task for the agent")
    use_tools: bool = Field(default=True, description="Whether to use tools")
    stream: bool = Field(default=False, description="Whether to stream the response")
    session_id: Optional[str] = Field(default=None, description="Session ID for context")
    max_iterations: Optional[int] = Field(default=10, description="Maximum iterations for agent execution")


class AgentResponse(BaseModel):
    """Response model for agent interactions."""

    query: str
    response: str
    reasoning: Optional[Dict[str, Any]] = None
    subagents_used: List[str] = []
    execution_time: float
    timestamp: str
    session_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    model_params: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    agent_ready: bool
    model: str
    available_subagents: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global coordinator, mcp_manager

    # Startup
    logger.info("Starting CoordinatorAgent API...")

    # Initialize telemetry if enabled
    tracer = None
    if settings.telemetry_enabled:
        logger.info(f"[TELEMETRY] Initializing {settings.telemetry_exporter} tracing...")

        langfuse_public = settings.langfuse_public_key or settings.telemetry_public_key or None
        langfuse_secret = settings.langfuse_secret_key or settings.telemetry_secret_key or None

        tracer = initialize_telemetry(
            service_name="coordinator-agent-api",
            exporter_type=settings.telemetry_exporter,
            otlp_endpoint=settings.telemetry_otlp_endpoint,
            jaeger_endpoint=settings.telemetry_jaeger_endpoint,
            langfuse_public_key=langfuse_public,
            langfuse_secret_key=langfuse_secret,
            langfuse_host=settings.langfuse_host,
            enabled=True
        )
        logger.info(f"[TELEMETRY] Tracing enabled with {settings.telemetry_exporter} exporter")
    else:
        logger.info("[TELEMETRY] Telemetry disabled")

    try:
        # Configure MCP servers
        servers = {
            "filesystem": MCPServerConfig(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            ),
            "time": MCPServerConfig(
                command="docker",
                args=["run", "-i", "--rm", "mcp/time"]
            ),
            "sequentialthinking": MCPServerConfig(
                command="docker",
                args=["run", "-i", "--rm", "mcp/sequentialthinking"]
            ),
            "memory": MCPServerConfig(
                command="docker",
                args=["run", "-i", "--rm", "-v", "/Users/udg/Projects/ai/agents/memory:/memory", "mcp/memory"]
            ),
            "vector-store": MCPServerConfig(
                command="python",
                args=["-m", "linus.mcp.vector_store"],
                cwd="/Users/udg/Projects/ai/agents/src",
                env={
                    "WV_HTTP_HOST": "localhost",
                    "WV_HTTP_PORT": "18080",
                    "WV_GRPC_HOST": "localhost",
                    "WV_GRPC_PORT": "50051",
                    "LLM_API_BASE": "http://localhost:11434/v1"
                }
            )
        }

        # Connect to MCP servers and get tools
        logger.info("[MCP] Connecting to MCP servers...")
        mcp_manager, mcp_tools = await connect_mcp_servers(servers)
        logger.info(f"[MCP] Connected. Total MCP tools: {len(mcp_tools)}")

        # Create tool mapping by server
        mcp_tools_by_server = {}
        for tool in mcp_tools:
            # Tool names are prefixed with server name (e.g., "filesystem_read_file")
            server_name = tool.name.split('_')[0]
            if server_name not in mcp_tools_by_server:
                mcp_tools_by_server[server_name] = []
            mcp_tools_by_server[server_name].append(tool)

        # Add EntitiesSearchTool
        entities_search_tool = EntitiesSearchTool()

        # Create Researcher subagent
        logger.info("[SUBAGENT] Creating Researcher agent...")
        researcher_tools = [entities_search_tool]
        # Add vector-store MCP tools
        if 'vector' in mcp_tools_by_server or 'vector-store' in mcp_tools_by_server:
            vector_tools = mcp_tools_by_server.get('vector-store', []) or mcp_tools_by_server.get('vector', [])
            researcher_tools.extend(vector_tools)
            logger.info(f"[SUBAGENT] Added {len(vector_tools)} vector-store tools to Researcher")

        researcher_agent = Agent(
            api_base=settings.llm_api_base,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            top_p=settings.llm_top_p,
            top_k=settings.llm_top_k,
            tools=researcher_tools,
            verbose=settings.agent_verbose,
            tracer=tracer,
            use_async=True,
            agent_name="Researcher"
        )
        logger.info(f"[SUBAGENT] Researcher created with {len(researcher_tools)} tools")

        # Create Jacksmith subagent (filesystem and time)
        logger.info("[SUBAGENT] Creating Jacksmith agent...")
        jacksmith_tools = []
        # Add filesystem tools
        if 'filesystem' in mcp_tools_by_server:
            jacksmith_tools.extend(mcp_tools_by_server['filesystem'])
        # Add time tools
        if 'time' in mcp_tools_by_server:
            jacksmith_tools.extend(mcp_tools_by_server['time'])

        jacksmith_agent = Agent(
            api_base=settings.llm_api_base,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            top_p=settings.llm_top_p,
            top_k=settings.llm_top_k,
            tools=jacksmith_tools,
            verbose=settings.agent_verbose,
            tracer=tracer,
            use_async=True,
            agent_name="Jacksmith"
        )
        logger.info(f"[SUBAGENT] Jacksmith created with {len(jacksmith_tools)} tools")

        # Create Reasoner subagent (sequential thinking and memory)
        logger.info("[SUBAGENT] Creating Reasoner agent...")
        reasoner_tools = []
        # Add sequential thinking tools
        if 'sequentialthinking' in mcp_tools_by_server:
            reasoner_tools.extend(mcp_tools_by_server['sequentialthinking'])
        # Add memory tools
        if 'memory' in mcp_tools_by_server:
            reasoner_tools.extend(mcp_tools_by_server['memory'])

        reasoner_agent = Agent(
            api_base=settings.llm_api_base,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            top_p=settings.llm_top_p,
            top_k=settings.llm_top_k,
            tools=reasoner_tools,
            verbose=settings.agent_verbose,
            tracer=tracer,
            use_async=True,
            agent_name="Reasoner"
        )
        logger.info(f"[SUBAGENT] Reasoner created with {len(reasoner_tools)} tools")

        # Create SubAgent wrappers
        subagents = [
            SubAgent(
                agent=researcher_agent,
                name="Researcher",
                description="Specializes in information retrieval, search, and data gathering using vector databases and entity search",
                capabilities=[
                    "Search for entities and related information",
                    "Query vector databases for semantic search",
                    "Retrieve relevant documents and content"
                ]
            ),
            SubAgent(
                agent=jacksmith_agent,
                name="Jacksmith",
                description="Handles file system operations, time-related queries, and document management",
                capabilities=[
                    "Read and write files",
                    "List directory contents",
                    "Get current time and date information",
                    "Manage temporary files"
                ]
            ),
            SubAgent(
                agent=reasoner_agent,
                name="Reasoner",
                description="Performs deep reasoning, maintains memory, and handles complex analytical tasks",
                capabilities=[
                    "Sequential step-by-step reasoning",
                    "Store and retrieve knowledge from memory",
                    "Analyze complex problems",
                    "Connect related concepts"
                ]
            )
        ]

        # Create CoordinatorAgent
        logger.info("[COORDINATOR] Creating CoordinatorAgent...")
        coordinator = CoordinatorAgent(
            llm=researcher_agent.llm,  # Use same LLM client
            model=settings.llm_model,
            subagents=subagents,
            verbose=settings.agent_verbose,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            top_p=settings.llm_top_p,
            top_k=settings.llm_top_k,
            api_base=settings.llm_api_base,
            max_iterations=15,
            logger=None,  # Use DI container
            telemetry=tracer,
            agent_name="Coordinator"
        )

        logger.info(f"[COORDINATOR] Initialized with {len(subagents)} subagents")
        logger.info(f"Using model: {settings.llm_model} at {settings.llm_api_base}")

    except Exception as e:
        logger.exception(f"Failed to initialize coordinator: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down CoordinatorAgent API...")

    # Cleanup MCP connections
    if mcp_manager:
        try:
            logger.info("Disconnecting MCP servers...")
            await mcp_manager.disconnect_all()
            logger.info("MCP servers disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting MCP servers: {e}")

    conversation_history.clear()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "architecture": "coordinator"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health status of the application."""
    global coordinator

    available_subagents = []
    if coordinator:
        available_subagents = [sa.name for sa in coordinator.subagents]

    return HealthResponse(
        status="healthy" if coordinator else "unhealthy",
        agent_ready=coordinator is not None,
        model=settings.llm_model,
        available_subagents=available_subagents
    )


@app.post("/agent/query", response_model=AgentResponse)
async def query_agent(request: AgentRequest):
    """Send a query to the coordinator agent and get a response."""
    global coordinator, conversation_history

    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    start_time = datetime.now()

    try:
        # Run the coordinator agent
        agent_response = await coordinator.run(request.query)

        execution_time = (datetime.now() - start_time).total_seconds()

        # Handle AgentResponse from coordinator
        from linus.agents.agent import AgentResponse as AgentResponseData

        if isinstance(agent_response, AgentResponseData):
            # Extract the result string from AgentResponse
            result_text = str(agent_response.result)

            # Extract subagents used from execution history
            subagents_used = []
            if agent_response.execution_history:
                subagents_used = [
                    item.get("subagent")
                    for item in agent_response.execution_history
                    if item.get("subagent")
                ]

            # Build reasoning info from execution history
            reasoning = None
            if agent_response.execution_history:
                reasoning = {
                    "completion_status": agent_response.completion_status,
                    "iterations": agent_response.metrics.total_iterations if agent_response.metrics else 0,
                    "execution_history": agent_response.execution_history
                }

            # Extract metrics
            metrics = agent_response.metrics.to_dict() if agent_response.metrics else None
        else:
            # Fallback for string response
            result_text = str(agent_response)
            subagents_used = []
            reasoning = None
            metrics = None

        # Extract model parameters
        model_params = {
            "base_url": coordinator.api_base,
            "model": coordinator.model,
            "temperature": coordinator.temperature,
            "max_tokens": coordinator.max_tokens,
            "top_p": coordinator.top_p,
            "top_k": coordinator.top_k
        }

        response = AgentResponse(
            query=request.query,
            response=result_text,
            reasoning=reasoning,
            subagents_used=list(set(subagents_used)),  # Remove duplicates
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            session_id=request.session_id,
            metrics=metrics,
            model_params=model_params
        )

        # Store in conversation history
        conversation_history.append(response.model_dump())

        # Limit history size
        if len(conversation_history) > 100:
            conversation_history = conversation_history[-100:]

        return response

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Agent query timed out after {settings.agent_timeout} seconds"
        )
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/subagents")
async def list_subagents():
    """List all available subagents and their capabilities."""
    global coordinator

    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    subagents_info = []
    for subagent in coordinator.subagents:
        # Get tool names from the subagent's agent
        tool_names = [tool.name for tool in subagent.agent.tools]

        subagent_info = {
            "name": subagent.name,
            "description": subagent.description,
            "capabilities": subagent.capabilities,
            "tools": tool_names,
            "tool_count": len(tool_names)
        }
        subagents_info.append(subagent_info)

    return {
        "subagents": subagents_info,
        "total_subagents": len(subagents_info)
    }


@app.get("/history")
async def get_history(limit: int = 10, session_id: Optional[str] = None):
    """Get conversation history."""
    history = conversation_history

    if session_id:
        history = [h for h in history if h.get("session_id") == session_id]

    # Return most recent items
    return {"history": history[-limit:], "total": len(history)}


@app.delete("/history")
async def clear_history():
    """Clear conversation history."""
    global conversation_history
    count = len(conversation_history)
    conversation_history.clear()
    return {"message": f"Cleared {count} conversation entries"}


@app.post("/agent/batch")
async def batch_queries(queries: List[str]):
    """Process multiple queries in batch."""
    global coordinator

    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    from linus.agents.agent import AgentResponse as AgentResponseData
    results = []

    for query in queries:
        try:
            agent_response = await coordinator.run(query)

            # Extract string result from AgentResponse
            if isinstance(agent_response, AgentResponseData):
                result_text = str(agent_response.result)
            else:
                result_text = str(agent_response)

            results.append({
                "query": query,
                "response": result_text,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "query": query,
                "error": str(e),
                "status": "failed"
            })

    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Logs directory: {os.path.abspath('logs')}")

    uvicorn.run(
        "app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
