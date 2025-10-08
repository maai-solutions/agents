"""FastAPI application to test the ReasoningAgent with Gemma3:27b."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import json
import asyncio
import os
import sys
from datetime import datetime
from loguru import logger

from linus.agents.agent import ReasoningAgent, create_gemma_agent, get_default_tools, create_custom_tool


# Configure logging early
os.makedirs("logs", exist_ok=True)

# Remove default handler and add custom ones
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/agent_api.log",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
    level="DEBUG"
)


class Settings(BaseSettings):
    """Application settings."""
    
    app_name: str = "ReasoningAgent API"
    app_version: str = "1.0.0"
    
    # Gemma/Ollama configuration
    llm_api_base: str = "http://localhost:11434/v1"
    llm_model: str = "gemma3:27b"
    llm_temperature: float = 0.7
    
    # Agent configuration
    agent_verbose: bool = True
    agent_timeout: int = 300  # 5 minutes timeout
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Global agent instance
agent: Optional[ReasoningAgent] = None

# Store conversation history
conversation_history: List[Dict[str, Any]] = []


class AgentRequest(BaseModel):
    """Request model for agent interactions."""
    
    query: str = Field(..., description="The user's query or task for the agent")
    use_tools: bool = Field(default=True, description="Whether to use tools")
    stream: bool = Field(default=False, description="Whether to stream the response")
    session_id: Optional[str] = Field(default=None, description="Session ID for context")


class AgentResponse(BaseModel):
    """Response model for agent interactions."""

    query: str
    response: str
    reasoning: Optional[Dict[str, Any]] = None
    tools_used: List[str] = []
    execution_time: float
    timestamp: str
    session_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class ToolTestRequest(BaseModel):
    """Request model for testing individual tools."""
    
    tool_name: str = Field(..., description="Name of the tool to test")
    tool_args: Dict[str, Any] = Field(..., description="Arguments for the tool")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    agent_ready: bool
    model: str
    available_tools: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global agent
    
    # Startup
    logger.info("Starting ReasoningAgent API...")
    
    try:
        # Initialize the agent
        tools = get_default_tools()
        
        # Add some custom tools for testing
        def get_time() -> str:
            """Get the current time."""
            return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        time_tool = create_custom_tool(
            name="get_time",
            description="Get the current date and time",
            func=get_time
        )
        tools.append(time_tool)
        
        agent = create_gemma_agent(
            api_base=settings.llm_api_base,
            model=settings.llm_model,
            tools=tools,
            verbose=settings.agent_verbose
        )
        
        logger.info(f"Agent initialized with {len(tools)} tools")
        logger.info(f"Using model: {settings.llm_model} at {settings.llm_api_base}")
        
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down ReasoningAgent API...")
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
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health status of the application."""
    global agent
    
    available_tools = []
    if agent:
        available_tools = [tool.name for tool in agent.tools]
    
    return HealthResponse(
        status="healthy" if agent else "unhealthy",
        agent_ready=agent is not None,
        model=settings.llm_model,
        available_tools=available_tools
    )


@app.post("/agent/query", response_model=AgentResponse)
async def query_agent(request: AgentRequest, background_tasks: BackgroundTasks):
    """Send a query to the agent and get a response."""
    global agent, conversation_history
    
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    start_time = datetime.now()
    
    try:
        # Run the agent (this is synchronous, so we run it in a thread pool)
        loop = asyncio.get_event_loop()
        agent_response = await loop.run_in_executor(None, agent.run, request.query)

        execution_time = (datetime.now() - start_time).total_seconds()

        # Handle AgentResponse from agent.run (returns AgentResponse by default with return_metrics=True)
        from linus.agents.agent import AgentResponse as AgentResponseData

        if isinstance(agent_response, AgentResponseData):
            # Extract the result string from AgentResponse
            result_text = str(agent_response.result)

            # Extract tools used from execution history
            tools_used = []
            if agent_response.execution_history:
                tools_used = [
                    item.get("tool")
                    for item in agent_response.execution_history
                    if item.get("tool")
                ]

            # Build reasoning info from execution history
            reasoning = None
            if agent_response.execution_history:
                reasoning = {
                    "completion_status": agent_response.completion_status,
                    "iterations": agent_response.metrics.total_iterations,
                    "execution_history": agent_response.execution_history
                }

            # Extract metrics
            metrics = agent_response.metrics.to_dict() if agent_response.metrics else None
        else:
            # Fallback for string response
            result_text = str(agent_response)
            tools_used = []
            reasoning = None
            metrics = None

        response = AgentResponse(
            query=request.query,
            response=result_text,
            reasoning=reasoning,
            tools_used=list(set(tools_used)),  # Remove duplicates
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            session_id=request.session_id,
            metrics=metrics
        )
        
        # Store in conversation history
        conversation_history.append(response.dict())
        
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
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/reasoning")
async def test_reasoning(request: AgentRequest):
    """Test only the reasoning phase without execution."""
    global agent
    
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Run reasoning phase only
        loop = asyncio.get_event_loop()
        reasoning_result = await loop.run_in_executor(
            None, 
            agent._reasoning_call, 
            request.query
        )
        
        return {
            "query": request.query,
            "has_sufficient_info": reasoning_result.has_sufficient_info,
            "reasoning": reasoning_result.reasoning,
            "planned_tasks": reasoning_result.tasks
        }
        
    except Exception as e:
        logger.error(f"Error in reasoning phase: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/test")
async def test_tool(request: ToolTestRequest):
    """Test a specific tool with given arguments."""
    global agent
    
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if request.tool_name not in agent.tool_map:
        available = list(agent.tool_map.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{request.tool_name}' not found. Available tools: {available}"
        )
    
    try:
        tool = agent.tool_map[request.tool_name]
        
        # Execute the tool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            tool.run,
            request.tool_args
        )
        
        return {
            "tool": request.tool_name,
            "args": request.tool_args,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error executing tool {request.tool_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools")
async def list_tools():
    """List all available tools."""
    global agent
    
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    tools_info = []
    for tool in agent.tools:
        tool_info = {
            "name": tool.name,
            "description": tool.description
        }
        
        # Add schema if available
        if hasattr(tool, 'args_schema') and tool.args_schema:
            tool_info["schema"] = tool.args_schema.schema()
        
        tools_info.append(tool_info)
    
    return {"tools": tools_info}


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
    global agent

    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    from linus.agents.agent import AgentResponse as AgentResponseData
    results = []

    for query in queries:
        try:
            loop = asyncio.get_event_loop()
            agent_response = await loop.run_in_executor(None, agent.run, query)

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


# Example endpoint for testing specific scenarios
@app.get("/test/scenarios")
async def test_scenarios():
    """Run predefined test scenarios."""
    global agent

    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    from linus.agents.agent import AgentResponse as AgentResponseData

    scenarios = [
        "What is the current time?",
        "Calculate 42 * 17 + 256",
        "Search for information about FastAPI",
        "First get the current time, then calculate 100/4, and finally search for Python"
    ]

    results = []
    for scenario in scenarios:
        try:
            loop = asyncio.get_event_loop()
            agent_response = await loop.run_in_executor(None, agent.run, scenario)

            # Extract string result from AgentResponse
            if isinstance(agent_response, AgentResponseData):
                result_text = str(agent_response.result)
            else:
                result_text = str(agent_response)

            results.append({
                "scenario": scenario,
                "result": result_text[:200] + "..." if len(result_text) > 200 else result_text
            })
        except Exception as e:
            results.append({
                "scenario": scenario,
                "error": str(e)
            })

    return {"test_results": results}


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