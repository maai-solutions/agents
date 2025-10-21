import asyncio
import json
from typing import Type, List, Optional, Dict, Any
from pydantic import BaseModel, Field
import weaviate
from weaviate.classes.query import MetadataQuery
from openai import OpenAI

from linus.agents.agent.tool_base import BaseTool
from linus.settings.settings import Settings


class VectorStoreInput(BaseModel):
    """Input for the vector store search tool."""
    query: str = Field(description="Natural language question or topic to search for in document content")

class VectorStoreTool(BaseTool):
    """Tool for searching document content/chunks using Weaviate vector store with hybrid search."""

    name: str = "vector_search"
    description: str = "Search for information in document content and text chunks. Use this to find WHAT was said, written, or documented about a topic. Returns full text content/paragraphs, not just entity names."
    args_schema: Type[BaseModel] = VectorStoreInput

    def __init__(self):
        """Initialize the VectorStoreTool with Weaviate client and settings."""
        super().__init__()
        self.settings = Settings()

        # Initialize Weaviate client
        self.client = weaviate.connect_to_custom(
            http_host=self.settings.wv_http_host,
            http_port=self.settings.wv_http_port,
            http_secure=(self.settings.wv_http_scheme == "https"),
            grpc_host=self.settings.wv_grpc_host,
            grpc_port=self.settings.wv_grpc_port,
            grpc_secure=(self.settings.wv_grpc_scheme == "https"),
        )

        # Initialize OpenAI client for embeddings
        self.embedding_client = OpenAI(
            base_url=self.settings.llm_api_base,
            api_key=self.settings.llm_api_key
        )

    def __del__(self):
        """Close Weaviate client connection."""
        if hasattr(self, 'client') and self.client:
            self.client.close()

    def embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text using OpenAI-compatible API.

        Args:
            text: Text to generate embedding for

        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.embedding_client.embeddings.create(
                model=self.settings.wv_embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Error generating embedding: {str(e)}")

    def hybrid_search(
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
            max_distance: Maximum distance for results (defaults to settings.wv_max_distance)
            alpha: Hybrid search parameter (0=keyword, 1=vector, defaults to settings.wv_alpha)
            limit: Maximum number of results (defaults to settings.wv_limit)

        Returns:
            JSON string with search results and citations
        """
        # Use settings defaults if not provided
        _collection = collection if collection is not None else self.settings.wv_collection
        _max_distance = max_distance if max_distance is not None else self.settings.wv_max_distance
        _alpha = alpha if alpha is not None else self.settings.wv_alpha
        _limit = limit if limit is not None else self.settings.wv_limit

        try:
            collection = self.client.collections.get(_collection)

            vector = self.embedding(query)

            # Perform hybrid search
            response = collection.query.hybrid(
                query=query,
                vector=vector,
                alpha=_alpha,
                limit=_limit,
                max_vector_distance=_max_distance,
                return_metadata=MetadataQuery(score=True)
            )

            # Format results with citations
            results = []
            citations = []

            for idx, obj in enumerate(response.objects, 1):
                score = obj.metadata.score
                content = obj.properties.get('text', '') or obj.properties.get('content', '')

                # Extract citation information from metadata
                document_id = obj.properties.get('document_id', obj.properties.get('doc_id', 'unknown'))
                chunk_number = obj.properties.get('chunk_number', obj.properties.get('chunk_id', idx))

                # Get other metadata (excluding text, content, tags, document_id, chunk_number)
                metadata = {
                    k: v for k, v in obj.properties.items()
                    if k not in ['text', 'content', 'tags', 'document_id', 'doc_id', 'chunk_number', 'chunk_id']
                }

                # Create result entry
                result_entry = {
                    "rank": idx,
                    "score": round(score, 4),
                    "content": content[:500] + ('...' if len(content) > 500 else ''),
                    "full_content": content,
                    "document_id": document_id,
                    "chunk_number": chunk_number,
                    "metadata": metadata
                }
                results.append(result_entry)

                # Create citation entry
                citation = {
                    "document_id": document_id,
                    "chunk_number": chunk_number,
                    "score": round(score, 4),
                    "content_preview": content[:200] + ('...' if len(content) > 200 else '')
                }
                citations.append(citation)

            if not results:
                return json.dumps({
                    "status": "no_results",
                    "message": f"No content found for query '{query}' within max_distance {_max_distance}",
                    "query": query,
                    "results": [],
                    "citations": []
                })

            # Return structured JSON response
            response_data = {
                "status": "success",
                "query": query,
                "search_params": {
                    "alpha": _alpha,
                    "limit": _limit,
                    "max_distance": _max_distance
                },
                "results": results,
                "citations": citations,
                "total_results": len(results)
            }

            return json.dumps(response_data, indent=2)

        except Exception as e:
            error_response = {
                "status": "error",
                "message": f"Error executing hybrid search: {str(e)}",
                "query": query,
                "results": [],
                "citations": []
            }
            return json.dumps(error_response)

    def _run(
        self,
        query: str
    ) -> str:
        """Execute the hybrid search.

        Args:
            query: Search query text
            
        Returns:
            Formatted search results
        """
        return self.hybrid_search(
            query, 
            collection=self.settings.wv_collection,
            max_distance=self.settings.wv_max_distance, 
            alpha=self.settings.wv_alpha, 
            limit=self.settings.wv_limit
        )

    async def _arun(self, query: str) -> str:
        """Execute the hybrid search asynchronously.

        Args:
            query: Search query text

        Returns:
            Formatted search results
        """
        # Run synchronous hybrid_search in a thread pool to avoid blocking
        return await asyncio.to_thread(
            self.hybrid_search,
            query,
            collection=self.settings.wv_collection,
            max_distance=self.settings.wv_max_distance,
            alpha=self.settings.wv_alpha,
            limit=self.settings.wv_limit
        )
