import os
import sys
from openai import OpenAI

from agentslm.core.agent.base import AgentBase
from dotenv import load_dotenv

from agentslm.core.interfaces.logger import ILogger
from agentslm.core.utils.logger import RichLogger
load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")


def get_logger() -> ILogger:
    """Helper to get a RichLogger instance."""
    return RichLogger(
        level="INFO",
        log_file="./logs/agent_tests.log",
        show_path=True,
        show_time=True,
        rich_tracebacks=True,
        use_stderr=False
    )

def test_simple_agent():
    """Test agent against a very simple question"""
    logger: ILogger = get_logger()
    
    llm = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        timeout=60,
        max_retries=3
    )
    agent = AgentBase[str,str](
        name="simple",
        llm = llm,
        model=MODEL_NAME,
        input_type=str,
        output_type=str,
        instructions="You are a helpful assistant.",
        logger=logger
    )

    input_data = "What is the capital of France?"
    output = agent.act(input_data)
    logger.info(f"Agent output: {output}")

    assert "Paris" in output

def test_structured_agent():
    logger: ILogger = get_logger()

    """Test agent with structured input and output using Pydantic models"""
    from pydantic import BaseModel

    class InputModel(BaseModel):
        question: str

    class OutputModel(BaseModel):
        answer: str

    llm = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        timeout=60,
        max_retries=3
    )
    agent = AgentBase[InputModel, OutputModel](
        name="structured",
        llm = llm,
        model=MODEL_NAME,
        input_type=InputModel,
        output_type=OutputModel,
        instructions="You are a helpful assistant that provides concise answers.",
        logger=logger
    )

    input_data = InputModel(question="What is the capital of Italy?")
    output = agent.act(input_data)
    logger.info(f"Agent output: {output}")
    assert isinstance(output, OutputModel)
    assert "Rome" in output.answer
    
    