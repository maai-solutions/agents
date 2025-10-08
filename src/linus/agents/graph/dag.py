"""DAG (Directed Acyclic Graph) for agent orchestration."""

from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import asyncio
from datetime import datetime
from loguru import logger

from .state import SharedState


class NodeStatus(Enum):
    """Status of a node in the DAG."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Edge:
    """Represents an edge between two nodes in the DAG."""
    from_node: str
    to_node: str
    condition: Optional[Callable[[SharedState], bool]] = None

    def can_traverse(self, state: SharedState) -> bool:
        """Check if this edge can be traversed.

        Args:
            state: Current shared state

        Returns:
            True if edge can be traversed
        """
        if self.condition is None:
            return True
        try:
            return self.condition(state)
        except Exception as e:
            logger.exception(f"[EDGE] Condition error {self.from_node}->{self.to_node}: {e}")
            return False


@dataclass
class AgentNode:
    """Represents a node (agent) in the DAG."""
    name: str
    agent: Any  # Agent instance
    description: str = ""
    input_mapping: Optional[Dict[str, str]] = None  # Map state keys to agent input
    output_key: Optional[str] = None  # Where to store agent output
    retry_count: int = 0
    timeout_seconds: Optional[float] = None
    on_error: str = "fail"  # "fail", "skip", "continue"

    # Runtime state
    status: NodeStatus = field(default=NodeStatus.PENDING, init=False)
    result: Optional[Any] = field(default=None, init=False)
    error: Optional[str] = field(default=None, init=False)
    start_time: Optional[datetime] = field(default=None, init=False)
    end_time: Optional[datetime] = field(default=None, init=False)
    execution_count: int = field(default=0, init=False)

    def prepare_input(self, state: SharedState) -> Any:
        """Prepare input for the agent from shared state.

        Args:
            state: Shared state

        Returns:
            Input for the agent
        """
        if self.input_mapping is None:
            # No mapping, return all state
            return state.get_all()

        # Map specific keys
        input_data = {}
        for agent_key, state_key in self.input_mapping.items():
            value = state.get(state_key)
            if value is not None:
                input_data[agent_key] = value

        logger.debug(f"[NODE:{self.name}] Prepared input: {list(input_data.keys())}")
        return input_data

    def execute(self, state: SharedState) -> Any:
        """Execute the agent node.

        Args:
            state: Shared state

        Returns:
            Agent execution result

        Raises:
            Exception: If execution fails and on_error="fail"
        """
        self.status = NodeStatus.RUNNING
        self.start_time = datetime.now()
        self.execution_count += 1

        logger.info(f"[NODE:{self.name}] Starting execution (attempt {self.execution_count})")

        try:
            # Prepare input
            input_data = self.prepare_input(state)

            # Execute agent with timeout if specified
            if self.timeout_seconds:
                # TODO: Implement timeout mechanism
                result = self.agent.run(input_data, return_metrics=False)
            else:
                result = self.agent.run(input_data, return_metrics=False)

            # Store result
            self.result = result
            self.status = NodeStatus.COMPLETED
            self.end_time = datetime.now()

            # Update shared state
            if self.output_key:
                state.set(self.output_key, result, source=self.name)

            execution_time = (self.end_time - self.start_time).total_seconds()
            logger.info(f"[NODE:{self.name}] Completed in {execution_time:.2f}s")

            return result

        except Exception as e:
            self.error = str(e)
            self.end_time = datetime.now()

            logger.exception(f"[NODE:{self.name}] Error: {e}")

            # Handle error based on configuration
            if self.on_error == "fail":
                self.status = NodeStatus.FAILED
                raise
            elif self.on_error == "skip":
                self.status = NodeStatus.SKIPPED
                logger.warning(f"[NODE:{self.name}] Skipping due to error")
                return None
            else:  # continue
                self.status = NodeStatus.COMPLETED
                logger.warning(f"[NODE:{self.name}] Continuing despite error")
                return None

    def reset(self) -> None:
        """Reset node to initial state."""
        self.status = NodeStatus.PENDING
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.execution_count = 0


class AgentDAG:
    """Directed Acyclic Graph for orchestrating multiple agents."""

    def __init__(self, name: str = "AgentDAG"):
        """Initialize the DAG.

        Args:
            name: DAG name
        """
        self.name = name
        self.nodes: Dict[str, AgentNode] = {}
        self.edges: List[Edge] = []
        self._start_nodes: Set[str] = set()
        self._end_nodes: Set[str] = set()

    def add_node(self, node: AgentNode) -> 'AgentDAG':
        """Add a node to the DAG.

        Args:
            node: Agent node to add

        Returns:
            Self for chaining
        """
        if node.name in self.nodes:
            raise ValueError(f"Node '{node.name}' already exists")

        self.nodes[node.name] = node
        logger.debug(f"[DAG:{self.name}] Added node '{node.name}'")

        return self

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        condition: Optional[Callable[[SharedState], bool]] = None
    ) -> 'AgentDAG':
        """Add an edge between two nodes.

        Args:
            from_node: Source node name
            to_node: Target node name
            condition: Optional condition function

        Returns:
            Self for chaining

        Raises:
            ValueError: If nodes don't exist or edge creates cycle
        """
        if from_node not in self.nodes:
            raise ValueError(f"Node '{from_node}' not found")
        if to_node not in self.nodes:
            raise ValueError(f"Node '{to_node}' not found")

        edge = Edge(from_node, to_node, condition)
        self.edges.append(edge)

        logger.debug(f"[DAG:{self.name}] Added edge {from_node} -> {to_node}")

        # Check for cycles
        if self._has_cycle():
            self.edges.remove(edge)
            raise ValueError(f"Adding edge {from_node}->{to_node} creates a cycle")

        return self

    def _has_cycle(self) -> bool:
        """Check if the DAG has any cycles using DFS.

        Returns:
            True if cycle detected
        """
        visited = set()
        rec_stack = set()

        def visit(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            # Visit all neighbors
            for edge in self.edges:
                if edge.from_node == node:
                    neighbor = edge.to_node
                    if neighbor not in visited:
                        if visit(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True

            rec_stack.remove(node)
            return False

        for node in self.nodes:
            if node not in visited:
                if visit(node):
                    return True

        return False

    def get_start_nodes(self) -> List[str]:
        """Get nodes with no incoming edges (entry points).

        Returns:
            List of start node names
        """
        has_incoming = {edge.to_node for edge in self.edges}
        return [name for name in self.nodes if name not in has_incoming]

    def get_end_nodes(self) -> List[str]:
        """Get nodes with no outgoing edges (exit points).

        Returns:
            List of end node names
        """
        has_outgoing = {edge.from_node for edge in self.edges}
        return [name for name in self.nodes if name not in has_outgoing]

    def get_dependencies(self, node_name: str) -> List[str]:
        """Get all nodes that must complete before this node.

        Args:
            node_name: Target node name

        Returns:
            List of dependency node names
        """
        return [edge.from_node for edge in self.edges if edge.to_node == node_name]

    def get_dependents(self, node_name: str) -> List[str]:
        """Get all nodes that depend on this node.

        Args:
            node_name: Source node name

        Returns:
            List of dependent node names
        """
        return [edge.to_node for edge in self.edges if edge.from_node == node_name]

    def validate(self) -> bool:
        """Validate the DAG structure.

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        # Check for cycles
        if self._has_cycle():
            raise ValueError("DAG contains cycles")

        # Check for disconnected nodes
        start_nodes = self.get_start_nodes()
        if not start_nodes:
            raise ValueError("No start nodes found (all nodes have incoming edges)")

        # Check that all nodes are reachable
        reachable = set()

        def mark_reachable(node: str):
            if node in reachable:
                return
            reachable.add(node)
            for dependent in self.get_dependents(node):
                mark_reachable(dependent)

        for start_node in start_nodes:
            mark_reachable(start_node)

        unreachable = set(self.nodes.keys()) - reachable
        if unreachable:
            logger.warning(f"[DAG:{self.name}] Unreachable nodes: {unreachable}")

        logger.info(f"[DAG:{self.name}] Validation passed")
        return True

    def reset(self) -> None:
        """Reset all nodes to initial state."""
        for node in self.nodes.values():
            node.reset()
        logger.info(f"[DAG:{self.name}] Reset all nodes")

    def get_execution_order(self) -> List[List[str]]:
        """Get topological execution order (levels that can run in parallel).

        Returns:
            List of lists, where each inner list contains nodes that can run in parallel
        """
        # Calculate in-degree for each node
        in_degree = {name: 0 for name in self.nodes}
        for edge in self.edges:
            in_degree[edge.to_node] += 1

        # Process nodes level by level
        levels = []
        processed = set()

        while len(processed) < len(self.nodes):
            # Find nodes with no remaining dependencies
            current_level = [
                name for name, degree in in_degree.items()
                if degree == 0 and name not in processed
            ]

            if not current_level:
                # This shouldn't happen if DAG is valid
                raise ValueError("Unable to determine execution order")

            levels.append(current_level)
            processed.update(current_level)

            # Update in-degrees
            for node_name in current_level:
                for dependent in self.get_dependents(node_name):
                    in_degree[dependent] -= 1

        return levels

    def visualize(self) -> str:
        """Generate a text visualization of the DAG.

        Returns:
            String representation of the DAG
        """
        lines = [f"DAG: {self.name}", "=" * 50]

        # Nodes
        lines.append("\nNodes:")
        for name, node in self.nodes.items():
            status_symbol = {
                NodeStatus.PENDING: "⏸",
                NodeStatus.RUNNING: "▶",
                NodeStatus.COMPLETED: "✓",
                NodeStatus.FAILED: "✗",
                NodeStatus.SKIPPED: "⊘"
            }.get(node.status, "?")

            lines.append(f"  {status_symbol} {name}: {node.description}")

        # Edges
        lines.append("\nEdges:")
        for edge in self.edges:
            cond = " [conditional]" if edge.condition else ""
            lines.append(f"  {edge.from_node} -> {edge.to_node}{cond}")

        # Execution order
        lines.append("\nExecution Order:")
        for i, level in enumerate(self.get_execution_order()):
            lines.append(f"  Level {i + 1}: {', '.join(level)}")

        return "\n".join(lines)


@dataclass
class ExecutionResult:
    """Result of DAG execution."""
    dag_name: str
    status: str  # "success", "partial", "failed"
    total_nodes: int
    completed_nodes: int
    failed_nodes: int
    skipped_nodes: int
    execution_time_seconds: float
    node_results: Dict[str, Any]
    errors: Dict[str, str]
    final_state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dag_name": self.dag_name,
            "status": self.status,
            "total_nodes": self.total_nodes,
            "completed_nodes": self.completed_nodes,
            "failed_nodes": self.failed_nodes,
            "skipped_nodes": self.skipped_nodes,
            "execution_time_seconds": round(self.execution_time_seconds, 3),
            "node_results": self.node_results,
            "errors": self.errors,
            "final_state": self.final_state
        }


class DAGExecutor:
    """Executor for running DAGs."""

    def __init__(self, dag: AgentDAG, state: Optional[SharedState] = None):
        """Initialize executor.

        Args:
            dag: DAG to execute
            state: Shared state (creates new if None)
        """
        self.dag = dag
        self.state = state or SharedState()

    def execute(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        parallel: bool = False
    ) -> ExecutionResult:
        """Execute the DAG.

        Args:
            initial_state: Initial state values
            parallel: Whether to run nodes in parallel (not implemented yet)

        Returns:
            Execution result
        """
        start_time = time.time()

        # Validate DAG
        self.dag.validate()

        # Reset all nodes
        self.dag.reset()

        # Initialize state
        if initial_state:
            for key, value in initial_state.items():
                self.state.set(key, value, source="initial")

        logger.info(f"[EXECUTOR] Starting DAG execution: {self.dag.name}")

        # Execute in topological order
        execution_order = self.dag.get_execution_order()

        for level_idx, level_nodes in enumerate(execution_order):
            logger.info(f"[EXECUTOR] Executing level {level_idx + 1}: {level_nodes}")

            for node_name in level_nodes:
                node = self.dag.nodes[node_name]

                # Check if all dependencies are satisfied
                deps = self.dag.get_dependencies(node_name)
                can_execute = True

                for dep_name in deps:
                    dep_node = self.dag.nodes[dep_name]

                    # Check edge condition
                    edge = next(
                        (e for e in self.dag.edges
                         if e.from_node == dep_name and e.to_node == node_name),
                        None
                    )

                    if dep_node.status == NodeStatus.FAILED:
                        logger.warning(
                            f"[EXECUTOR] Skipping {node_name}: dependency {dep_name} failed"
                        )
                        node.status = NodeStatus.SKIPPED
                        can_execute = False
                        break

                    if edge and not edge.can_traverse(self.state):
                        logger.info(
                            f"[EXECUTOR] Skipping {node_name}: edge condition not met"
                        )
                        node.status = NodeStatus.SKIPPED
                        can_execute = False
                        break

                if can_execute:
                    try:
                        node.execute(self.state)
                    except Exception as e:
                        logger.exception(f"[EXECUTOR] Node {node_name} failed: {e}")
                        if node.status != NodeStatus.SKIPPED:
                            node.status = NodeStatus.FAILED

        # Collect results
        execution_time = time.time() - start_time

        node_results = {}
        errors = {}
        completed = 0
        failed = 0
        skipped = 0

        for name, node in self.dag.nodes.items():
            if node.status == NodeStatus.COMPLETED:
                completed += 1
                node_results[name] = node.result
            elif node.status == NodeStatus.FAILED:
                failed += 1
                errors[name] = node.error or "Unknown error"
            elif node.status == NodeStatus.SKIPPED:
                skipped += 1

        # Determine overall status
        if failed > 0:
            status = "failed"
        elif skipped > 0:
            status = "partial"
        else:
            status = "success"

        result = ExecutionResult(
            dag_name=self.dag.name,
            status=status,
            total_nodes=len(self.dag.nodes),
            completed_nodes=completed,
            failed_nodes=failed,
            skipped_nodes=skipped,
            execution_time_seconds=execution_time,
            node_results=node_results,
            errors=errors,
            final_state=self.state.get_all()
        )

        logger.info(
            f"[EXECUTOR] Completed: {completed}/{len(self.dag.nodes)} nodes, "
            f"status={status}, time={execution_time:.2f}s"
        )

        return result

    async def aexecute(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        parallel: bool = True
    ) -> ExecutionResult:
        """Async version: Execute the DAG with optional parallel execution.

        Args:
            initial_state: Initial state values
            parallel: Whether to run nodes in parallel (default: True)

        Returns:
            Execution result
        """
        start_time = time.time()

        # Validate DAG
        self.dag.validate()

        # Reset all nodes
        self.dag.reset()

        # Initialize state
        if initial_state:
            for key, value in initial_state.items():
                self.state.set(key, value, source="initial")

        logger.info(f"[ASYNC-EXECUTOR] Starting DAG execution: {self.dag.name} (parallel={parallel})")

        # Execute in topological order
        execution_order = self.dag.get_execution_order()

        for level_idx, level_nodes in enumerate(execution_order):
            logger.info(f"[ASYNC-EXECUTOR] Executing level {level_idx + 1}: {level_nodes}")

            if parallel and len(level_nodes) > 1:
                # Execute nodes in this level in parallel
                logger.info(f"[ASYNC-EXECUTOR] Running {len(level_nodes)} nodes in parallel")
                await self._execute_level_parallel(level_nodes)
            else:
                # Execute nodes sequentially
                for node_name in level_nodes:
                    await self._execute_node_async(node_name)

        # Collect results
        execution_time = time.time() - start_time

        node_results = {}
        errors = {}
        completed = 0
        failed = 0
        skipped = 0

        for name, node in self.dag.nodes.items():
            if node.status == NodeStatus.COMPLETED:
                completed += 1
                node_results[name] = node.result
            elif node.status == NodeStatus.FAILED:
                failed += 1
                errors[name] = node.error or "Unknown error"
            elif node.status == NodeStatus.SKIPPED:
                skipped += 1

        # Determine overall status
        if failed > 0:
            status = "failed"
        elif skipped > 0:
            status = "partial"
        else:
            status = "success"

        result = ExecutionResult(
            dag_name=self.dag.name,
            status=status,
            total_nodes=len(self.dag.nodes),
            completed_nodes=completed,
            failed_nodes=failed,
            skipped_nodes=skipped,
            execution_time_seconds=execution_time,
            node_results=node_results,
            errors=errors,
            final_state=self.state.get_all()
        )

        logger.info(
            f"[ASYNC-EXECUTOR] Completed: {completed}/{len(self.dag.nodes)} nodes, "
            f"status={status}, time={execution_time:.2f}s"
        )

        return result

    async def _execute_level_parallel(self, level_nodes: List[str]) -> None:
        """Execute all nodes in a level in parallel.

        Args:
            level_nodes: List of node names to execute in parallel
        """
        # Create tasks for all nodes in this level
        tasks = []
        for node_name in level_nodes:
            task = asyncio.create_task(self._execute_node_async(node_name))
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_node_async(self, node_name: str) -> None:
        """Execute a single node asynchronously.

        Args:
            node_name: Name of the node to execute
        """
        node = self.dag.nodes[node_name]

        # Check if all dependencies are satisfied
        deps = self.dag.get_dependencies(node_name)
        can_execute = True

        for dep_name in deps:
            dep_node = self.dag.nodes[dep_name]

            # Check edge condition
            edge = next(
                (e for e in self.dag.edges
                 if e.from_node == dep_name and e.to_node == node_name),
                None
            )

            if dep_node.status == NodeStatus.FAILED:
                logger.warning(
                    f"[ASYNC-EXECUTOR] Skipping {node_name}: dependency {dep_name} failed"
                )
                node.status = NodeStatus.SKIPPED
                can_execute = False
                break

            if edge and not edge.can_traverse(self.state):
                logger.info(
                    f"[ASYNC-EXECUTOR] Skipping {node_name}: edge condition not met"
                )
                node.status = NodeStatus.SKIPPED
                can_execute = False
                break

        if can_execute:
            try:
                # Check if agent has async method
                agent = node.agent
                if hasattr(agent, 'arun'):
                    # Use async method
                    await self._execute_node_with_arun(node)
                else:
                    # Fall back to sync execution in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, node.execute, self.state)
            except Exception as e:
                logger.exception(f"[ASYNC-EXECUTOR] Node {node_name} failed: {e}")
                if node.status != NodeStatus.SKIPPED:
                    node.status = NodeStatus.FAILED

    async def _execute_node_with_arun(self, node: 'AgentNode') -> None:
        """Execute a node using agent's async arun method.

        Args:
            node: The node to execute
        """
        try:
            node.status = NodeStatus.RUNNING
            logger.info(f"[ASYNC-EXECUTOR] Executing node '{node.name}' (async)")

            # Prepare input
            if node.input_mapping:
                agent_input = {}
                for agent_key, state_key in node.input_mapping.items():
                    value = self.state.get(state_key)
                    if value is not None:
                        agent_input[agent_key] = value

                if agent_input:
                    input_text = " ".join([f"{k}: {v}" for k, v in agent_input.items()])
                else:
                    input_text = "Process the current state"
            else:
                # Pass all state as context
                all_state = self.state.get_all()
                input_text = f"Current state: {all_state}"

            # Execute agent with async method
            result = await node.agent.arun(input_text, return_metrics=False)

            # Store result
            node.result = result
            node.status = NodeStatus.COMPLETED

            # Save to state if output_key specified
            if node.output_key:
                self.state.set(node.output_key, result, source=node.name)
                logger.debug(f"[ASYNC-EXECUTOR] Saved result to state['{node.output_key}']")

            logger.info(f"[ASYNC-EXECUTOR] Node '{node.name}' completed (async)")

        except Exception as e:
            logger.exception(f"[ASYNC-EXECUTOR] Node '{node.name}' failed: {e}", exc_info=True)
            node.status = NodeStatus.FAILED
            node.error = str(e)

            if node.on_error == "fail":
                raise
            elif node.on_error == "skip":
                logger.warning(f"[ASYNC-EXECUTOR] Skipping failed node '{node.name}'")
            # on_error == "continue" just logs and continues
