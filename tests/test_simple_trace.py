"""Simple test to check if tracer is working."""

import os
from dotenv import load_dotenv
from loguru import logger

from linus.agents.telemetry import initialize_telemetry

# Load environment variables
load_dotenv()

def test_tracer_init():
    """Test basic tracer initialization."""

    logger.info("Initializing Langfuse tracer...")

    # Initialize Langfuse tracer
    tracer = initialize_telemetry(
        service_name="test-simple",
        exporter_type="langfuse",
        session_id="test-123",
        enabled=True
    )

    logger.info(f"Tracer initialized: {type(tracer)}")
    logger.info(f"Tracer enabled: {tracer.enabled}")
    logger.info(f"Tracer client: {type(tracer.client) if hasattr(tracer, 'client') else 'No client'}")

    # Check if we can access client methods
    if hasattr(tracer, 'client'):
        logger.info(f"Client methods: {[m for m in dir(tracer.client) if not m.startswith('_')][:10]}")

    logger.info("✅ Tracer initialization successful!")

if __name__ == "__main__":
    test_tracer_init()
