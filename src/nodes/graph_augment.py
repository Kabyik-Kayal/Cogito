"""
Graph Augment Node

Expands retrieval context using the NetworkX document graph
after performing Vector Search
"""
import sys
from typing import List
from src.state import GraphState
from src.db.graph_store import GraphStore
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

class GraphAugmentNode:
    """
    Graph-based context expansion node.
    Args:
        depth: How many graph hops to traverse (1 = immediate neighbors)
        max_neighbors_per_node: Limit neighbors to avoid context explosion
    """
    def __init__(self, depth: int=1, max_neighbors_per_node: int=3):
        """
        Initialize the graph augment node.
        Args:
            depth: How many graph hops to traverse (1=immediate neighbors)
            max_neighbors_per_node: Limit neighbors to avoid context explosion
        """
    
        self.depth = depth
        self.max_neighbors_per_node = max_neighbors_per_node

        # Load graph from disk
        self.graph_store = GraphStore()
        try:
            self.graph_store.load()
            logger.info(f"GraphAugmentNode initialized (depth={depth}, max_neighbors={max_neighbors_per_node})")
        except FileNotFoundError:
            logger.error("Graph file not found! Run ingestion pipeline first.")
            raise CustomException("Graph not found. Run ingestion first.", sys)

    def __call__(self, state: GraphState) -> GraphState:
        """
        Execute graph extension.
        Args:
            state: Current graph state
        Returns:
            Updated state with graph_augmented_docs populated
        """
        try:
            document_ids = state.document_ids
            logger.info(f"Expanding context for {len(document_ids)} retrieved nodes")

            if not document_ids:
                logger.warning("No document IDs to expand!")
                state.graph_augmented_docs = []
                return state

            # Collect all neighbor nodes
            all_neighbor_ids = set()
            for node_id in document_ids:
                # Get neighbors from graph
                neighbors = self.graph_store.get_neighbors(node_id=node_id, depth=self.depth)
                # Limit neighbors per node
                neighbors = neighbors[:self.max_neighbors_per_node]
                logger.info(f" Node {node_id}: found {len(neighbors)}")
                all_neighbor_ids.update(neighbors)
            
            # Remove nodes that were already retrieved (avoid duplication)
            all_neighbor_ids = all_neighbor_ids - set(document_ids)

            # Get content for all neighbor nodes
            augmented_docs = []
            for neighbor_id in all_neighbor_ids:
                content = self.graph_store.get_node_content(neighbor_id)
                if content:
                    augmented_docs.append(content)
            
            logger.info(f" Added {len(augmented_docs)} graph-augmented documents")
            # Update state
            state.graph_augmented_docs = augmented_docs
            return state
        
        except Exception as e:
            raise CustomException(f"Graph augmentation failed: {e}", sys)
    
    def graph_augment_node(state: GraphState) -> GraphState:
        """
        Standalone graph augmentation function for LangGraph.

        Args:
            state: Current graph state
        Returns:
            Updated state with graph-augmented documents
        """
        augmenter = GraphAugmentNode(depth=1, max_neighbors_per_node=3)
        return augmenter(state)