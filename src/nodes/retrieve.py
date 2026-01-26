"""
Retrieve Node

Performs semantic search in ChromaDB to find relevant documents
based on the user's question.
"""
import sys
from typing import Dict, Any
from src.state import GraphState
from src.db.vector_store import VectorStore
from utils.logger import get_logger
from utils.custom_exception import CustomException 

logger = get_logger(__name__)

class RetrieveNode:
    """
    Vector Search node

    Input: state.question
    Output: state.documents, state.document_ids

    This node queries ChromaDB using semantic similarity
    to find the top-k most relevant document chunks.
    """
    def __init__(
        self,
        collection_name: str = "cogito_docs",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 3
    ):
        """
        Initialize the retrieve node.

        Args:
            collection_name: ChromaDB collection name
            embedding_model: Embedding model (must match ingestion)
            top_k: Number of documents to retrieve
        """
        self.top_k = top_k
        self.vector_store = VectorStore(collection_name=collection_name,
                                        embedding_model=embedding_model)
        logger.info(f"RetrieveNode initialized (top_k={top_k})")

    def __call__(self, state: GraphState) -> GraphState:
        """
        Execute retrieval
        Args:
            state: Current graph state
        Returns:
            Updated state with documents and document_ids populated
        """
        try:
            question = state.question
            logger.info(f"Retrieving documents for: '{question}'")

            # Perform vector search
            results = self.vector_store.search(query=question, top_k=self.top_k)
            
            # Extract results
            documents = results['documents']
            node_ids = results['node_ids']
            distances = results['distances']

            # Log retrieval results
            logger.info(f"Retrieved {len(documents)} documents")
            for i, (node_id, distance) in enumerate(zip(node_ids, distances)):
                logger.info(f" [{i+1}] Node: {node_id}, Distance: {distance:.4f}")
            
            # Update state
            state.documents = documents
            state.document_ids = node_ids

            # Check if retrieval found anything
            if not documents:
                logger.warning("No documents retrieved! RAG will likely fail.")
            return state
        
        except Exception as e:
            raise CustomException(f"Retrieval failed: {e}", sys)

def retrieve_node(state: GraphState) -> GraphState:
    """
    Standalone retrieval function for LangGraph.

    Args:
        state: Current Graph state
    Returns:
        Updated state with retrieved documents
    """
    retriever = RetrieveNode(
        collection_name = "cogito_docs",
        top_k=3
    )
    return retriever(state)

# Test the retrieve node
if __name__ == "__main__":
    print("Testing RetrieveNode...\n")
    
    # Check if vector store has data
    try:
        vs = VectorStore(collection_name="test_ingestion")
        stats = vs.get_collection_stats()
        print(f"Vector store stats: {stats}\n")
        
        if stats['total_documents'] == 0:
            print(" Warning: Vector store is empty!")
            print("Run the ingestion pipeline first:")
            print("  python -m src.ingestion.pipeline\n")
            exit(1)
        
        # Create test state
        from src.state import create_initial_state
        
        test_question = "How do you use Python as a calculator?"
        state = create_initial_state(test_question)
        
        print(f"Test Question: {test_question}")
        print("="*60)
        
        # Initialize and run retrieve node
        retriever = RetrieveNode(
            collection_name="test_ingestion",
            top_k=3
        )
        
        updated_state = retriever(state)
        
        # Display results
        print(f"\nRetrieved {len(updated_state.documents)} documents:\n")
        
        for i, (doc, node_id) in enumerate(zip(
            updated_state.documents,
            updated_state.document_ids
        )):
            print(f"{i+1}. Node ID: {node_id}")
            print(f"   Content: {doc[:150]}...")
            print()
        
        print("✓ RetrieveNode test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure you've run the ingestion pipeline first:")
        print("python -m src.ingestion.pipeline")