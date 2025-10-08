from typing import Type, List, Optional
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
            http_host=self.settings.wv_url,
            http_port=self.settings.wv_port,
            http_secure=(self.settings.wv_scheme == "https"),
            grpc_host=self.settings.wv_url,
            grpc_port=50051,
            grpc_secure=(self.settings.wv_scheme == "https"),
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
            Formatted string with search results
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

            # Filter by max_distance and format results
            results = []
            for idx, obj in enumerate(response.objects, 1):
                score = obj.metadata.score
                content = obj.properties.get('text', '') or obj.properties.get('content', '')
                metadata = {k: v for k, v in obj.properties.items() if k not in ['text', 'content', 'tags']}

                result_str = f"{idx}. [Score: {score:.4f}]\n"
                result_str += f"   Content: {content[:500]}{'...' if len(content) > 500 else ''}\n"
                if metadata:
                    result_str += f"   Metadata: {metadata}\n"
                results.append(result_str)

            if not results:
                return f"No content found for query '{query}' within max_distance {_max_distance}"

            header = f"Content search results for '{query}' (alpha={_alpha}, limit={_limit}, max_distance={_max_distance}):\n\n"
            return header + "\n".join(results)

        except Exception as e:
            return f"Error executing hybrid search: {str(e)}"

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

    async def _arun(self, *args, **kwargs):
        """Async version not implemented."""
        raise NotImplementedError("Async vector search not supported")
