"""MCP Server for Vector Store operations using Weaviate.

This module implements a Model Context Protocol (MCP) server that exposes
vector store search capabilities through the MCP protocol.
"""

import asyncio
import json
from typing import Any, List, Optional

import weaviate
from loguru import logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from openai import OpenAI
from pydantic import Field
from rich.console import Console
from weaviate.classes.query import MetadataQuery

from linus.agents.logging_config import setup_rich_logging, log_with_panel, log_metrics
from linus.settings.settings import Settings


class VectorStoreMCPServer:
    """MCP Server for vector store operations."""

    def __init__(self):
        """Initialize the MCP server with Weaviate client and settings."""
        # Setup logging - use stderr for MCP to avoid interfering with JSON-RPC on stdout
        self.console = setup_rich_logging(
            level="INFO",
            log_file="logs/vector_store_mcp.log",
            show_path=True,
            show_time=True,
            use_stderr=True  # Critical: MCP requires stdout for JSON-RPC only
        )

        logger.info("Initializing VectorStoreMCPServer")

        try:
            self.settings = Settings()
            logger.debug(f"Settings loaded: wv_host={self.settings.wv_http_host}:{self.settings.wv_http_port}")

            self.server = Server("vector-store-server")
            logger.info("MCP Server instance created")

            # Initialize Weaviate client
            logger.info(f"Connecting to Weaviate at {self.settings.wv_http_host}:{self.settings.wv_http_port}")
            self.weaviate_client = weaviate.connect_to_custom(
                http_host=self.settings.wv_http_host,
                http_port=self.settings.wv_http_port,
                http_secure=(self.settings.wv_http_scheme == "https"),
                grpc_host=self.settings.wv_grpc_host,
                grpc_port=self.settings.wv_grpc_port,
                grpc_secure=(self.settings.wv_grpc_scheme == "https"),
            )
            logger.info("✓ Weaviate client connected successfully")

            # Initialize OpenAI client for embeddings
            logger.info(f"Initializing embedding client: {self.settings.llm_api_base} with model {self.settings.wv_embedding_model}")
            self.embedding_client = OpenAI(
                base_url=self.settings.llm_api_base,
                api_key=self.settings.llm_api_key
            )
            logger.info("✓ Embedding client initialized successfully")

            # Register handlers
            self._register_handlers()
            logger.info("✓ MCP handlers registered")

            log_with_panel(
                f"[green]VectorStoreMCPServer initialized successfully[/green]\n"
                f"Weaviate: {self.settings.wv_http_host}:{self.settings.wv_http_port}\n"
                f"Collection: {self.settings.wv_collection}\n"
                f"Embedding Model: {self.settings.wv_embedding_model}",
                title="MCP Vector Store Server",
                console=self.console,
                border_style="green"
            )

        except Exception as e:
            logger.exception(f"Failed to initialize VectorStoreMCPServer: {e}")
            raise

    def _register_handlers(self):
        """Register MCP protocol handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools."""
            logger.debug("list_tools() called")
            tools = [
                Tool(
                    name="vector_search",
                    description=(
                        "Search for information in document content and text chunks using hybrid search. "
                        "Use this to find WHAT was said, written, or documented about a topic. "
                        "Returns full text content/paragraphs, not just entity names."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language question or topic to search for in document content"
                            },
                            "collection": {
                                "type": "string",
                                "description": f"Weaviate collection name (default: {self.settings.wv_collection})"
                            },
                            "max_distance": {
                                "type": "number",
                                "description": f"Maximum distance for results (default: {self.settings.wv_max_distance})"
                            },
                            "alpha": {
                                "type": "number",
                                "description": f"Hybrid search parameter (0=keyword, 1=vector, default: {self.settings.wv_alpha})"
                            },
                            "limit": {
                                "type": "integer",
                                "description": f"Maximum number of results (default: {self.settings.wv_limit})"
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
            logger.info(f"Returning {len(tools)} available tool(s)")
            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> list[TextContent]:
            """Handle tool calls."""
            logger.info(f"Tool called: {name}")
            logger.debug(f"Tool arguments: {arguments}")

            if name == "vector_search":
                try:
                    result = await self._vector_search(
                        query=arguments.get("query"),
                        collection=arguments.get("collection"),
                        max_distance=arguments.get("max_distance"),
                        alpha=arguments.get("alpha"),
                        limit=arguments.get("limit")
                    )
                    logger.info(f"Tool '{name}' executed successfully")
                    return [TextContent(type="text", text=result)]
                except Exception as e:
                    logger.error(f"Tool '{name}' execution failed: {e}")
                    raise
            else:
                logger.error(f"Unknown tool requested: {name}")
                raise ValueError(f"Unknown tool: {name}")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text using OpenAI-compatible API.

        Args:
            text: Text to generate embedding for

        Returns:
            List of floats representing the embedding vector
        """
        logger.debug(f"Generating embedding for text (length={len(text)})")

        try:
            response = self.embedding_client.embeddings.create(
                model=self.settings.wv_embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            logger.debug(f"Embedding generated successfully (dimension={len(embedding)})")
            return embedding

        except Exception as e:
            logger.exception(f"Error generating embedding: {e}")
            raise RuntimeError(f"Error generating embedding: {str(e)}")

    async def _vector_search(
        self,
        query: str,
        collection: Optional[str] = None,
        max_distance: Optional[float] = None,
        alpha: Optional[float] = None,
        limit: Optional[int] = None
    ) -> str:
        """Perform hybrid search combining vector and keyword search.

        Args:
            query: Search query text
            collection: Weaviate collection name
            max_distance: Maximum distance for results
            alpha: Hybrid search parameter (0=keyword, 1=vector)
            limit: Maximum number of results

        Returns:
            Formatted string with search results
        """
        # Use settings defaults if not provided
        _collection = collection if collection is not None else self.settings.wv_collection
        _max_distance = max_distance if max_distance is not None else self.settings.wv_max_distance
        _alpha = alpha if alpha is not None else self.settings.wv_alpha
        _limit = limit if limit is not None else self.settings.wv_limit

        logger.info(f"Starting vector search: query='{query[:50]}...', collection={_collection}, alpha={_alpha}, limit={_limit}")

        try:
            # Get collection
            logger.debug(f"Getting collection: {_collection}")
            collection_obj = self.weaviate_client.collections.get(_collection)

            # Generate embedding in thread pool to avoid blocking
            logger.debug("Generating query embedding")
            loop = asyncio.get_event_loop()
            vector = await loop.run_in_executor(None, self._generate_embedding, query)

            # Perform hybrid search
            logger.debug(f"Executing hybrid search (alpha={_alpha}, limit={_limit}, max_distance={_max_distance})")
            response = collection_obj.query.hybrid(
                query=query,
                vector=vector,
                alpha=_alpha,
                limit=_limit,
                max_vector_distance=_max_distance,
                return_metadata=MetadataQuery(score=True)
            )

            # Format results and remove duplicates based on text content
            results = []
            seen_texts = set()  # Track unique text content to avoid duplicates
            num_results = len(response.objects)
            unique_results = 0
            logger.info(f"Found {num_results} result(s)")

            for idx, obj in enumerate(response.objects, 1):
                score = obj.metadata.score
                content = obj.properties.get('text', '') or obj.properties.get('content', '')
                metadata = {k: v for k, v in obj.properties.items() if k not in ['text', 'content', 'tags']}

                # Skip duplicate content - compare normalized text (stripped and lowercased)
                normalized_content = content.strip().lower()
                if normalized_content in seen_texts:
                    logger.debug(f"Skipping duplicate content at position {idx} (score={score:.4f})")
                    continue
                
                # Add to seen texts to prevent future duplicates
                seen_texts.add(normalized_content)
                unique_results += 1

                logger.debug(f"Result {unique_results}: score={score:.4f}, content_length={len(content)}")

                result_str = f"{unique_results}. [Score: {score:.4f}]\n"
                result_str += f"   Content: {content[:500]}{'...' if len(content) > 500 else ''}\n"
                if metadata:
                    result_str += f"   Metadata: {metadata}\n"
                results.append(result_str)

            if not results:
                logger.warning(f"No results found for query '{query}' within max_distance {_max_distance}")
                return f"No content found for query '{query}' within max_distance {_max_distance}"

            # Log search metrics with deduplication info
            search_metrics = {
                "query_length": len(query),
                "results_found": num_results,
                "unique_results": unique_results,
                "duplicates_removed": num_results - unique_results,
                "collection": _collection,
                "alpha": _alpha,
                "max_distance": _max_distance,
                "limit": _limit
            }
            log_metrics(search_metrics, title="Vector Search Metrics", console=self.console)

            logger.info(f"✓ Vector search completed successfully with {unique_results} unique result(s) from {num_results} total (removed {num_results - unique_results} duplicates)")

            header = f"Content search results for '{query}' (alpha={_alpha}, limit={_limit}, max_distance={_max_distance}):\n"
            header += f"Showing {unique_results} unique results from {num_results} total (removed {num_results - unique_results} duplicates)\n\n"
            return header + "\n".join(results)

        except Exception as e:
            logger.exception(f"Error executing hybrid search: {e}")
            error_msg = f"Error executing hybrid search: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def run(self):
        """Run the MCP server."""
        logger.info("Starting MCP server with stdio transport")

        try:
            async with stdio_server() as (read_stream, write_stream):
                logger.info("✓ MCP server running, waiting for requests...")
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )
        except Exception as e:
            logger.exception(f"Error running MCP server: {e}")
            raise
        finally:
            logger.info("MCP server shutdown")

    def __del__(self):
        """Close Weaviate client connection."""
        if hasattr(self, 'weaviate_client') and self.weaviate_client:
            logger.debug("Closing Weaviate client connection")
            try:
                self.weaviate_client.close()
                logger.info("✓ Weaviate client connection closed")
            except Exception as e:
                logger.error(f"Error closing Weaviate client: {e}")


async def main():
    """Main entry point for the MCP server."""
    logger.info("=" * 60)
    logger.info("Starting VectorStore MCP Server")
    logger.info("=" * 60)

    try:
        server = VectorStoreMCPServer()
        await server.run()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.exception(f"Server failed with error: {e}")
        raise
    finally:
        logger.info("Server stopped")


if __name__ == "__main__":
    asyncio.run(main())
