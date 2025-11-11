"""Swarm - Multi-agent coordination and task handoff system.

The Swarm class enables multiple agents to work together, with support for:
- Agent registration and management
- Task handoff between agents
- Intent detection for routing
- Shared context and state
"""

from typing import Dict, List, Optional, Union, Any
import re
from pydantic import BaseModel

from .light_agent import LightAgent
from ..graph.state import SharedState
from ..di import ILogger, get_container


class AgentHandoff(BaseModel):
    """Represents a handoff from one agent to another."""
    from_agent: str
    to_agent: str
    context: str
    reason: Optional[str] = None


class Swarm:
    """Multi-agent coordination system.

    The Swarm manages multiple agents and enables them to:
    - Work together on complex tasks
    - Hand off tasks to specialized agents
    - Share context and state
    - Route queries based on intent detection

    Example:
        ```python
        # Create specialized agents
        researcher = LightAgent(
            llm=llm,
            model="gemma3:27b",
            tools=[SearchTool()],
            instructions="You are a research specialist.",
            agent_name="researcher"
        )

        calculator = LightAgent(
            llm=llm,
            model="gemma3:27b",
            tools=[CalculatorTool()],
            instructions="You are a math specialist.",
            agent_name="calculator"
        )

        # Create swarm
        swarm = Swarm()
        swarm.register(researcher, calculator)

        # Run query - swarm will route to appropriate agent
        result = await swarm.run("Calculate 42 * 17")
        ```
    """

    def __init__(
        self,
        state: Optional[SharedState] = None,
        enable_intent_detection: bool = True,
        logger: Optional[ILogger] = None
    ):
        """Initialize the Swarm.

        Args:
            state: Optional shared state for all agents
            enable_intent_detection: Whether to enable automatic intent detection
            logger: Optional logger instance
        """
        self.agents: Dict[str, LightAgent] = {}
        self.state = state or SharedState()
        self.enable_intent_detection = enable_intent_detection

        # Dependency injection for logger
        container = get_container()
        self.logger = logger or container.get_logger()

        # Track handoffs
        self.handoff_history: List[AgentHandoff] = []

    def register(self, *agents: LightAgent) -> "Swarm":
        """Register one or more agents with the swarm.

        Args:
            *agents: Variable number of LightAgent instances

        Returns:
            Self for method chaining

        Example:
            swarm.register(agent1, agent2, agent3)
        """
        for agent in agents:
            if agent.agent_name in self.agents:
                self.logger.warning(f"[SWARM] Agent '{agent.agent_name}' already registered, replacing")

            self.agents[agent.agent_name] = agent

            # Set swarm reference in agent
            agent.swarm = self

            # Share state with agent
            agent.state = self.state

            self.logger.info(f"[SWARM] Registered agent: {agent.agent_name}")

        return self

    def unregister(self, agent_name: str) -> bool:
        """Unregister an agent from the swarm.

        Args:
            agent_name: Name of the agent to unregister

        Returns:
            True if agent was unregistered, False if not found
        """
        if agent_name in self.agents:
            agent = self.agents[agent_name]
            agent.swarm = None
            del self.agents[agent_name]
            self.logger.info(f"[SWARM] Unregistered agent: {agent_name}")
            return True

        self.logger.warning(f"[SWARM] Agent '{agent_name}' not found for unregistration")
        return False

    def get_agent(self, agent_name: str) -> Optional[LightAgent]:
        """Get an agent by name.

        Args:
            agent_name: Name of the agent

        Returns:
            LightAgent instance or None if not found
        """
        return self.agents.get(agent_name)

    def list_agents(self) -> List[str]:
        """Get list of all registered agent names.

        Returns:
            List of agent names
        """
        return list(self.agents.keys())

    def get_agent_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all agents and their capabilities.

        Returns:
            Dictionary mapping agent names to their information
        """
        capabilities = {}

        for name, agent in self.agents.items():
            capabilities[name] = {
                "name": name,
                "instructions": agent.instructions,
                "role": agent.role,
                "tools": [tool.name for tool in agent.tools],
                "model": agent.model
            }

        return capabilities

    async def run(
        self,
        query: str,
        agent_name: Optional[str] = None,
        return_metrics: bool = True
    ) -> Any:
        """Run a query through the swarm.

        If agent_name is specified, routes directly to that agent.
        Otherwise, uses intent detection to route to the best agent.

        Args:
            query: The user's query
            agent_name: Optional specific agent to use
            return_metrics: Whether to return metrics

        Returns:
            Agent response

        Raises:
            ValueError: If specified agent not found or no agents registered
        """
        if not self.agents:
            raise ValueError("No agents registered in swarm")

        # Direct routing if agent specified
        if agent_name:
            if agent_name not in self.agents:
                raise ValueError(f"Agent '{agent_name}' not found in swarm")

            self.logger.info(f"[SWARM] Routing query to specified agent: {agent_name}")
            agent = self.agents[agent_name]
            return await agent.run(query, return_metrics=return_metrics)

        # Intent detection routing
        if self.enable_intent_detection and len(self.agents) > 1:
            target_agent_name = await self._detect_intent(query)
            if target_agent_name and target_agent_name in self.agents:
                self.logger.info(f"[SWARM] Intent detection routed to: {target_agent_name}")
                agent = self.agents[target_agent_name]
                return await agent.run(query, return_metrics=return_metrics)

        # Fallback: use first agent
        first_agent_name = list(self.agents.keys())[0]
        self.logger.info(f"[SWARM] Using default agent: {first_agent_name}")
        agent = self.agents[first_agent_name]
        return await agent.run(query, return_metrics=return_metrics)

    async def _detect_intent(self, query: str) -> Optional[str]:
        """Detect which agent should handle the query.

        Uses the first agent's LLM to analyze the query and determine
        which agent is best suited based on their capabilities.

        Args:
            query: The user's query

        Returns:
            Name of the target agent or None if unclear
        """
        # Get capabilities of all agents
        capabilities = self.get_agent_capabilities()

        # Build prompt for intent detection
        agent_descriptions = []
        for name, info in capabilities.items():
            desc = f"- {name}: {info['instructions']}"
            if info['role']:
                desc += f" Role: {info['role']}"
            desc += f" Tools: {', '.join(info['tools'])}"
            agent_descriptions.append(desc)

        prompt = f"""Analyze the following user query and determine which agent should handle it.

Available agents:
{chr(10).join(agent_descriptions)}

User query: {query}

Respond with ONLY the agent name that should handle this query, or "unclear" if you cannot determine.
Agent name:"""

        # Use the first agent's LLM for intent detection
        first_agent = list(self.agents.values())[0]

        try:
            messages = [
                {"role": "system", "content": "You are an intent detection system. Output only the agent name."},
                {"role": "user", "content": prompt}
            ]

            response = await first_agent.llm.chat.completions.create(
                model=first_agent.model,
                messages=messages,
                temperature=0.3,  # Low temperature for consistent routing
                max_tokens=50
            )

            intent = response.choices[0].message.content.strip().lower()
            self.logger.debug(f"[SWARM] Intent detection result: {intent}")

            # Check if intent matches any agent name
            for agent_name in self.agents.keys():
                if agent_name.lower() in intent:
                    return agent_name

            return None

        except Exception as e:
            self.logger.exception(f"[SWARM] Intent detection error: {e}")
            return None

    async def handoff(
        self,
        from_agent: str,
        to_agent: str,
        context: str,
        reason: Optional[str] = None
    ) -> Any:
        """Hand off a task from one agent to another.

        Args:
            from_agent: Name of the agent handing off
            to_agent: Name of the agent receiving the task
            context: Context to pass to the receiving agent
            reason: Optional reason for the handoff

        Returns:
            Response from the receiving agent

        Raises:
            ValueError: If either agent not found
        """
        if from_agent not in self.agents:
            raise ValueError(f"Source agent '{from_agent}' not found")

        if to_agent not in self.agents:
            raise ValueError(f"Target agent '{to_agent}' not found")

        self.logger.info(f"[SWARM] Handoff: {from_agent} -> {to_agent}")
        if reason:
            self.logger.info(f"[SWARM] Handoff reason: {reason}")

        # Track handoff
        handoff = AgentHandoff(
            from_agent=from_agent,
            to_agent=to_agent,
            context=context,
            reason=reason
        )
        self.handoff_history.append(handoff)

        # Execute on target agent
        target_agent = self.agents[to_agent]
        return await target_agent.run(context, return_metrics=True)

    def get_handoff_history(self) -> List[AgentHandoff]:
        """Get the history of all handoffs in this swarm.

        Returns:
            List of AgentHandoff instances
        """
        return self.handoff_history.copy()

    def clear_handoff_history(self):
        """Clear the handoff history."""
        self.handoff_history.clear()

    def get_shared_state(self) -> SharedState:
        """Get the shared state object.

        Returns:
            SharedState instance
        """
        return self.state

    def __len__(self) -> int:
        """Get the number of registered agents."""
        return len(self.agents)

    def __contains__(self, agent_name: str) -> bool:
        """Check if an agent is registered.

        Args:
            agent_name: Name of the agent

        Returns:
            True if agent is registered, False otherwise
        """
        return agent_name in self.agents

    def __repr__(self) -> str:
        """String representation of the swarm."""
        agent_list = ", ".join(self.agents.keys())
        return f"Swarm(agents=[{agent_list}], count={len(self.agents)})"
