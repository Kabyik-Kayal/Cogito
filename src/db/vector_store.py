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
from utils.gpu_selector import get_device
import sys
import gc
import torch

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
            # Sanitize collection name for ChromaDB requirements
            # Rules: 3-512 chars, [a-zA-Z0-9._-], start/end with alphanumeric
            import re
            sanitized_name = re.sub(r'[^a-zA-Z0-9._-]', '_', collection_name)  # Replace invalid chars
            sanitized_name = re.sub(r'^[^a-zA-Z0-9]+', '', sanitized_name)  # Remove leading non-alphanumeric
            sanitized_name = re.sub(r'[^a-zA-Z0-9]+$', '', sanitized_name)  # Remove trailing non-alphanumeric
            sanitized_name = sanitized_name[:512] if len(sanitized_name) > 512 else sanitized_name
            if len(sanitized_name) < 3:
                sanitized_name = sanitized_name + "_collection"
            
            logger.info(f"Initializing ChromaDB VectorStore with collection: {sanitized_name}")
            
            #Initialize ChromaDB client (persistent storage)
            self.client = chromadb.PersistentClient(
                path=str(CHROMA_DB_DIR),
                settings=Settings(anonymized_telemetry=False)
            )

            # Auto-detect device (MPS, CUDA, XPU, or CPU)
            self.device, self.gpu = get_device()
            logger.info(f"Loading embedding model: {embedding_model} on device: {self.device}")
            self.embedding_model = SentenceTransformer(embedding_model, device=self.device)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                    name=sanitized_name,
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
            
            # Add to ChromaDB in batches (max batch size is ~5000)
            batch_size = 5000
            total_added = 0
            
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                
                self.collection.add(
                    documents=documents[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    ids=node_ids[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )
                
                total_added += (batch_end - i)
                logger.info(f"Added batch {i//batch_size + 1}: {total_added}/{len(documents)} documents")

            logger.info(f"Added {len(documents)} documents. Total:{self.collection.count()}")

        except Exception as e:
            raise CustomException(f"Failed to add documents: {e}", sys)
    
    def unload_model(self) -> None:
        """
        Unload the embedding model from memory to free GPU/MPS resources.
        Call this after batch embedding operations are complete.
        """
        try:
            if hasattr(self, 'embedding_model') and self.embedding_model is not None:
                logger.info("Unloading embedding model from memory...")
                del self.embedding_model
                self.embedding_model = None
                
                # Clear GPU memory based on device type
                if hasattr(self, 'device'):
                    if self.device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    elif self.device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif self.device == "xpu" and hasattr(torch, 'xpu') and torch.xpu.is_available():
                        torch.xpu.empty_cache()
                gc.collect()
                
                logger.info("Embedding model unloaded, memory freed")
        except Exception as e:
            logger.warning(f"Failed to unload model: {e}")

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
    
    
    def delete_collection(self) -> None:
        """
        Permanently delete the collection from ChromaDB (does not recreate).
        """
        try:
            collection_name = self.collection.name
            logger.warning(f"Permanently deleting collection: {collection_name}")
            self.client.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' deleted permanently")
        except Exception as e:
            raise CustomException(f"Failed to delete collection: {e}", sys)