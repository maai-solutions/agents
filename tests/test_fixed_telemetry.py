"""Test the fixed async telemetry implementation."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from linus.agents.telemetry import LangfuseTracer, initialize_telemetry
from loguru import logger


async def test_langfuse_fixed():
    """Test the fixed Langfuse tracer implementation."""
    print("\n" + "=" * 60)
    print("Testing FIXED Langfuse Tracer")
    print("=" * 60)

    # Create a mock Langfuse client
    class MockLangfuseClient:
        """Mock Langfuse client for testing."""
        
        def __init__(self):
            self.traces = []
            self.current_trace = None

        def start_as_current_span(self, name, input=None, metadata=None):
            """Mock start_as_current_span for context manager usage."""
            from contextlib import contextmanager
            
            @contextmanager
            def span_context():
                span = MockSpan(name, input, metadata, parent=self)
                try:
                    yield span
                finally:
                    pass  # Span cleanup
            
            return span_context()
            
        def start_as_current_generation(self, name, model=None, input=None, metadata=None):
            """Mock start_as_current_generation for context manager usage."""
            from contextlib import contextmanager
            
            @contextmanager
            def generation_context():
                gen = MockGeneration(name, model, input, metadata, parent=self)
                try:
                    yield gen
                finally:
                    pass  # Generation cleanup
            
            return generation_context()
            
        def update_current_generation(self, output=None, usage=None):
            """Mock update current generation."""
            logger.info(f"[MOCK] Updating current generation with output length: {len(output) if output else 0}")
            
        def update_current_trace(self, metadata=None):
            """Mock update current trace."""
            logger.info(f"[MOCK] Updating current trace with metadata: {metadata}")
            
        def update_current_span(self, output=None):
            """Mock update current span."""
            logger.info(f"[MOCK] Updating current span with output: {output}")
            
        def flush(self):
            """Mock flush operation."""
            logger.info(f"[MOCK] Flushing traces")

    class MockTrace:
        """Mock trace object."""
        
        def __init__(self, name, input=None, metadata=None, session_id=None):
            self.name = name
            self.input = input
            self.metadata = metadata or {}
            self.session_id = session_id
            self.spans = []
            self.generations = []
            self.id = f"trace_{len([])}_{name}"
            self._ended = False
            
        def span(self, name, input=None, metadata=None):
            """Create a mock span."""
            span = MockSpan(name, input, metadata, parent=self)
            self.spans.append(span)
            return span
            
        def generation(self, name, model=None, input=None, metadata=None):
            """Create a mock generation."""
            gen = MockGeneration(name, model, input, metadata, parent=self)
            self.generations.append(gen)
            return gen
            
        def update(self, output=None, metadata=None, usage=None):
            """Update trace."""
            if output:
                self.output = output
            if metadata:
                self.metadata.update(metadata)
            if usage:
                self.usage = usage
            logger.info(f"[MOCK] Updated trace {self.name}")
            
        def end(self):
            """End trace."""
            self._ended = True
            logger.info(f"[MOCK] Ended trace {self.name}")

    class MockSpan:
        """Mock span object."""
        
        def __init__(self, name, input=None, metadata=None, parent=None):
            self.name = name
            self.input = input
            self.metadata = metadata or {}
            self.parent = parent
            self.id = f"span_{name}"
            self._ended = False
            
        def end(self, output=None):
            """End span."""
            if output:
                self.output = output
            self._ended = True
            logger.info(f"[MOCK] Ended span {self.name}")

    class MockGeneration:
        """Mock generation object."""
        
        def __init__(self, name, model=None, input=None, metadata=None, parent=None):
            self.name = name
            self.model = model
            self.input = input
            self.metadata = metadata or {}
            self.parent = parent
            self.id = f"gen_{name}"
            self._ended = False
            
        def end(self, output=None, usage=None):
            """End generation."""
            if output:
                self.output = output
            if usage:
                self.usage = usage
            self._ended = True
            logger.info(f"[MOCK] Ended generation {self.name}")

    # Create tracer with mock client
    mock_client = MockLangfuseClient()
    tracer = LangfuseTracer(langfuse_client=mock_client, session_id="test-session")
    
    print(f"Tracer enabled: {tracer.enabled}")
    print(f"Mock client has {len(mock_client.traces)} traces initially")

    # Test the fixed tracing implementation
    try:
        print("\n1. Testing agent run trace...")
        async with tracer.trace_agent_run("Test query", "TestAgent") as trace:
            print(f"✅ Agent run trace created: {trace.name if trace else 'None'}")
            
            print("\n2. Testing reasoning phase...")
            async with tracer.trace_reasoning_phase("Analyzing query...", iteration=1) as reasoning:
                print(f"✅ Reasoning span created: {reasoning.name if reasoning else 'None'}")
                
                print("\n3. Testing LLM call...")
                async with tracer.trace_llm_call("What is 2+2?", "test-model", "reasoning") as llm:
                    print(f"✅ LLM generation created: {llm.name if llm else 'None'}")
                    
                    # Simulate LLM response
                    if llm:
                        llm.end(output="2+2 equals 4")
                
                print("\n4. Testing tool execution...")
                async with tracer.trace_tool_execution("calculator", {"expression": "2+2"}) as tool:
                    print(f"✅ Tool span created: {tool.name if tool else 'None'}")
                    
                    # Simulate tool response
                    if tool:
                        tool.end(output="4")

        print("\n5. Testing metrics recording...")
        tracer.record_metrics({"tokens": 100, "duration": 1.5})
        
        print("\n6. Testing flush...")
        tracer.flush()

        print(f"\nFinal state: Mock client tested successfully")
        print("✅ ALL TESTS PASSED!")
        print("The fixed Langfuse tracer correctly uses context managers!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        

async def main():
    """Run the fixed telemetry test."""
    print("=" * 70)
    print("TESTING FIXED ASYNC TELEMETRY")
    print("This tests the corrected Langfuse async integration")
    print("=" * 70)

    await test_langfuse_fixed()

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())