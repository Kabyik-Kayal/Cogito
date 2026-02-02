"""
Graph Store Module using NetworkX

Manages the document relationship graph that captures structural connections
between documentation chunks (parent-child sections, hyperlinks, references).

This solves the "missing context" problem by expanding retrieval beyond just vector similarity.
"""

import networkx as nx
import pickle
import json
from typing import List, Dict, Any, Optional, Set 
from pathlib import Path 

from config.paths import GRAPH_PICKLE_PATH, GRAPH_METADATA_PATH, GRAPH_STORE_DIR
from utils.logger import get_logger
from utils.custom_exception import CustomException
import sys

logger = get_logger(__name__)

class GraphStore:
    """
    Manages a NetworkX graph representing document structure.

    Nodes: Individual documentation chunks (sections, paragraphs)
    Edges: Relationships (parent-child, hyperlinks, references)

    Node attributes:
        - node_id: Unique Identifier
        - content: Text content
        - section_type: Type of section (header, code, paragraph)
        - metadata: Additional info (source, url, etc.)
    
    Edge attributes:
        - relationship_type: Type of connection (parent_child, hyperlink, reference)
    """

    def __init__(self, collection_name: str = "default", auto_load: bool = True):
        """
        Initialize the graph store for a specific collection.
        
        Args:
            collection_name: Name of the collection (creates separate graph file per collection)
            auto_load: If True, attempt to load existing graph from disk.
                      If no saved graph exists, creates an empty graph.
        """
        # Sanitize collection name for file system compatibility
        import re
        sanitized_name = re.sub(r'[^a-zA-Z0-9._-]', '_', collection_name)
        sanitized_name = re.sub(r'^[^a-zA-Z0-9]+', '', sanitized_name)
        sanitized_name = re.sub(r'[^a-zA-Z0-9]+$', '', sanitized_name)
        if len(sanitized_name) < 3:
            sanitized_name = sanitized_name + "_collection"
        
        self.collection_name = sanitized_name
        self.graph = nx.DiGraph()
        
        # Compute collection-specific paths
        self.graph_path = GRAPH_STORE_DIR / f"graph_{sanitized_name}.pkl"
        self.metadata_path = GRAPH_STORE_DIR / f"graph_{sanitized_name}_metadata.json"
        
        if auto_load and self.graph_path.exists():
            try:
                self.load()
            except Exception as e:
                logger.warning(f"Could not load saved graph for '{collection_name}': {e}. Starting with empty graph.")
                self.graph = nx.DiGraph()
                logger.info(f"GraphStore initialized with an empty graph for collection '{collection_name}'")
        else:
            logger.info(f"GraphStore initialized with an empty graph for collection '{collection_name}'")
    
    def add_node(self, node_id: str, content: str,
                section_type: str="paragraph", metadata: Optional[Dict[str, Any]]=None):
        """
        Add a node to the graph.

        Args:
            node_id: Unique identifier for the node
            content: Text content of the document chunk
            section_type: Type of section (header, paragraph, code, table)
            metadata: Additional metadata (source, url, parent_section, etc.)
        """
        try: 
            if metadata is None:
                metadata={}
            self.graph.add_node(node_id, content=content, section_type=section_type, metadata=metadata)
        except Exception as e:
            raise CustomException(f"Failed to add node {node_id}: {e}", sys)
        
    def add_edge(self, source_id: str, target_id: str, relationship_type: str="reference") -> None:
        """
        Add an edge between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Type of relationship (parent_child, hyperlink, reference)
        """
        try:
            if source_id not in self.graph:
                logger.warning(f"Source node {source_id} not in graph, skipping edge")
                return
            
            if target_id not in self.graph:
                logger.warning(f"Target node {target_id} not in graph, skipping edge")
                return
            
            self.graph.add_edge(source_id, target_id, relationship_type=relationship_type)
        
        except Exception as e:
            raise CustomException(f"Failed to add edge {source_id}->{target_id}: {e}", sys)
    
    def get_neighbors(self, node_id: str, depth: int=1,
                    relationship_filter: Optional[List[str]] = None) -> List[str]:
        """
        Get neighboring node IDs within a specified depth.
        This is used by GraphAugmentNode to expand retrieval context.
        
        Args:
            node_id: Starting node ID
            depth: How many hops away to search (1 = immediate neighbors)
            relationship_filter: Only include edges with these relationship types
        
        Returns:
            List of neighboring node IDs
        """
        try:
            if node_id not in self.graph:
                logger.warning(f"Node {node_id} not found in graph")
                return []
            neighbors = set()

            # BFS to find neighbors within depth
            current_level = {node_id}
            visited = {node_id}

            for _ in range(depth):
                next_level = set()
                for node in current_level:
                    # Get successors and predecessors
                    for neighbor in list(self.graph.successors(node)) + list(self.graph.predecessors(node)):
                        if neighbor not in visited:
                            if relationship_filter:
                                edge_data = self.graph.get_edge_data(node, neighbor) or self.graph.get_edge_data(neighbor,node)
                                if edge_data and edge_data.get('relationship_type') in relationship_filter:
                                    neighbors.add(neighbor)
                                    next_level.add(neighbor)
                                    visited.add(neighbor)
                            else:
                                neighbors.add(neighbor)
                                next_level.add(neighbor)
                                visited.add(neighbor)
                current_level = next_level
            return list(neighbors)
    
        except Exception as e:
            raise CustomException(f"Failed to get neighbors for {node_id}: {e}", sys)
    
    def get_node_content(self, node_id: str) -> Optional[str]:
        """
        Get the text content of a node.
        """
        if node_id in self.graph:
            return self.graph.nodes[node_id].get('content', '')
        return None
    
    def save(self, path: Optional[Path] = None) -> None:
        """
        Save the graph to disk as a pickle file.

        Arg:
            path: Custom save path (defaults to collection-specific path)
        """
        try:
            save_path = path or self.graph_path
            GRAPH_STORE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving graph to {save_path}")
            with open(save_path, 'wb') as f:
                pickle.dump(self.graph, f)
            
            # Save metadata as JSON for human readability
            metadata = {
                "collection": self.collection_name,
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges(),
                "node_types": self._get_node_type_counts()
            }

            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Graph saved: {metadata['num_nodes']} nodes, {metadata['num_edges']} edges")
        
        except Exception as e:
            raise CustomException(f"Failed to save graph: {e}", sys)
    
    def load(self, path: Optional[Path]=None) -> None:
        """
        Load the graph from the disk.

        Args:
            path: Custom load path (defaults to collection-specific path)
        """
        try:
            load_path = path or self.graph_path

            if not load_path.exists():
                raise FileNotFoundError(f"Graph file not found: {load_path}")
            
            logger.info(f"Loading graph from {load_path}")

            with open(load_path, 'rb') as f:
                self.graph = pickle.load(f)

            logger.info(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        except Exception as e:
            raise CustomException(f"Failed to load graph: {e}", sys)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Graph.
        """
        return {
            "num_nodes":self.graph.number_of_nodes(),
            "num_edges":self.graph.number_of_edges(),
            "node_types":self._get_node_type_counts(),
            "is_directed":self.graph.is_directed(),
            "available_degree": sum(dict(self.graph.degree()).values())/max(self.graph.number_of_nodes(),1)
        }
    
    def delete_graph(self) -> None:
        """
        Permanently delete the graph files from disk.
        """
        try:
            deleted = []
            
            if self.graph_path.exists():
                self.graph_path.unlink()
                deleted.append("graph pickle")
                logger.info(f"Deleted graph file: {self.graph_path}")
            
            if self.metadata_path.exists():
                self.metadata_path.unlink()
                deleted.append("metadata")
                logger.info(f"Deleted graph metadata: {self.metadata_path}")
            
            # Clear in-memory graph
            self.graph.clear()
            
            logger.info(f"Graph '{self.collection_name}' deleted permanently")
        except Exception as e:
            raise CustomException(f"Failed to delete graph: {e}", sys)

    def _get_node_type_counts(self) -> Dict[str,int]:
        type_counts = {}
        for node_id in self.graph.nodes():
            node_type = self.graph.nodes[node_id].get('section_type','unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts