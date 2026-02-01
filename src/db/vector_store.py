"""
Vector Store Module using ChromaDB

Handles document embeddings, storage, and semantic search.
Each document is stored with its content, embeddings and metadata
including the NetworkX node_id for graph expansion.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from config.paths import CHROMA_DB_DIR
from utils.logger import get_logger
from utils.custom_exception import CustomException
import sys

logger = get_logger(__name__)

class VectorStore:
    """
    Wrapper for ChromaDB operations.

    Handles document embedding, storage, and retrieval using semantic search.
    Uses sentence-transformers for local embedding generation.
    """
    
    def __init__(self, collection_name: str="cogito_docs", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store.

        Args:
        collection_name: Name of the ChromaDB collection
        embedding_model: HuggingFace model for embeddings (small, fast model)
        """
        try:
            logger.info(f"Initializing ChromaDB VectorStore with collection: {collection_name}")
            
            #Initialize ChromaDB client (persistent storage)
            self.client = chromadb.PersistentClient(
                path=str(CHROMA_DB_DIR),
                settings=Settings(anonymized_telemetry=False)
            )

            # Load embedding model (Locally)
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model, device="cpu")

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "Cogito RAG document store"}
                )
            logger.info(f"Vector initialized. Documents in collections:{self.collection.count()}")

        except Exception as e:
            raise CustomException(f"Failed to initialize VectorStore: {e}", sys)
    
    def add_documents(
        self,
        documents: List[str],
        node_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document text contents
            node_ids: List of NetworkX node IDs, matching document lengths
            metadatas: Optional list of metadata dicts for each document
        """
        try:
            if len(documents) != len(node_ids):
                raise ValueError("documents and node_ids must have same length")
            logger.info(f"Addind {len(documents)} documents to vector store...")
        
            # Generate embeddings locally
            embeddings = self.embedding_model.encode(
                documents,
                show_progress_bar=True,
                convert_to_numpy=True
            ).tolist()

            # Prepare metadata (include node_id for graph expansion)
            if metadatas is None:
                metadatas = [{"node_id": node_id} for node_id in node_ids]
            else:
                for i, meta in enumerate(metadatas):
                    meta["node_id"] = node_ids[i]
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=node_ids,
                metadatas=metadatas
            )

            logger.info(f"Added {len(documents)} documents. Total:{self.collection.count()}")

        except Exception as e:
            raise CustomException(f"Failed to add documents: {e}", sys)

    def search(self, query:str, top_k:int=3) -> Dict[str, Any]:
        """
        Semantic search for relevant documents.

        Args:
            query: The searh query
            top_k: Number of results to return

        Returns:
            Dict containing:
                - documents: List of document texts
                - node_ids: List of node IDs (for graph expansion)
                - metadatas: List of metadata dicts
                - distances: List of similarity scores
        """
        try:
            logger.info(f"Searching for: '{query}' (top_k={top_k})")

            # Generate query embedding
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).tolist()

            # Query ChromaDB
            results = self.collection.query(query_embeddings=[query_embedding],
                                            n_results=top_k,
                                            include=["documents","metadatas","distances"])
            
            # Extract results (ChromaDB returns nested lists)
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            # Extra node_ids from metadata
            node_ids = [meta.get("node_id", "") for meta in metadatas]
            
            logger.info(f"Found {len(documents)} documents")

            return {"documents":documents,
                    "node_ids":node_ids,
                    "metadatas":metadatas,
                    "distances":distances}
        except Exception as e:
            raise CustomException(f"Search failed: {e}", sys)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get Statistics about the Collection.
        """
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name,
            "embedding_model": self.embedding_model.__class__.__name__
        }
    
    def clear_collection(self) -> None:
        """
        Delete all documents from the Collection.
        """
        try:
            logger.warning(f"Clearing collection: {self.collection.name}")
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"description":"Cogito RAG document store"}
            )
            logger.info("Collection Cleared")
        except Exception as e:
            raise CustomException(f"Failed to clear collection: {e}", sys)