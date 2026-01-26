"""
Ingestion Pipeline Module 

Orchestrates the complete data ingestion process:
1. Scrape documentation (DocumentationScraper)
2. Build relationship graph (GraphStore)
3. Store embeddings (VectorStore)

Converting raw docs into structured knowledge.
"""
from typing import List, Optional
from pathlib import Path 
import json

from src.ingestion.scraper import DocumentationScraper, DocumentNode
from src.db.graph_store import GraphStore
from src.db.vector_store import VectorStore
from config.paths import DATA_RAW_DIR, DATA_PROCESSED_DIR
from utils.logger import get_logger
from utils.custom_exception import CustomException
import sys

logger = get_logger(__name__)

class IngestionPipeline:
    """
    End-to-end ingestion pipeline.

    Takes a documention URL and produces:
    1. NetworkX graph with document relationships
    2. ChromaDB collection with embeddings
    3. Metadata JSON for inspection
    """

    def __init__(self, collection_name:str="cogito_docs", embedding_model:str="all-MiniLM-L6-v2"):
        """
        Initialize the pipeline.

        Args:
            collection_name: Name for ChromaDB collection
            embedding_model: HuggingFace model for embeddings
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Initialize stores
        self.graph_store = GraphStore()
        self.vector_store = VectorStore(collection_name=collection_name, embedding_model=embedding_model)
        logger.info("Ingestion Pipeline initialized")
    
    def run(self, base_url: str, max_pages: int=20, delay: float=2.0) -> dict:
        """
        Run the complete ingestion pipeline.

        Args:
            base_url: Starting URL for documentation scraping
            max_pages: Maximum pages to scrape
            delay: Delay between requests (seconds)
        
        Returns:
            Dictionary with pipeline statistics
        """
        try:
            logger.info("Starting Ingestion Pipeline")
            
            # Step 1: Scrape documentation
            logger.info("Scraping documentations")
            nodes = self._scrape_docs(base_url, max_pages, delay)
            logger.info("Documents scraped successfully")

            # Step 2: Build graph structure
            logger.info("Building document graphs")
            self._build_graph(nodes)
            logger.info("Document graphs built successfully")

            # Step 3: Store in Vector database
            logger.info("Creating embeddings and storing in ChromaDB")
            self._store_vectors(nodes)
            logger.info("Embeddings successfully created and stored in ChromaDB")

            # Step 4: Save Metadata
            logger.info("Saving metadata")
            stats = self._save_metadata(base_url, nodes)
            logger.info("Metadata saved successfully")

            logger.info("Ingestion Pipeline Completed Successfully")
            logger.info(f"Total nodes: {stats['total_nodes']}")
            logger.info(f"Graph edges: {stats['graph_edges']}")
            logger.info(f"Vector DB docs: {stats['vector_db_count']}")

            return stats
    
        except Exception as e:
            raise CustomException(f"Ingestion pipeline failed: {e}", sys)
    
    def _scrape_docs(self, base_url: str, max_pages: int, delay: float) -> List[DocumentNode]:
        """Step 1: Scrape Documentation."""
        scraper = DocumentationScraper(base_url=base_url, max_pages=max_pages, delay=delay)
        nodes = scraper.scrape()
        logger.info(f"Scraped {len(nodes)} nodes from {len(scraper.visited_urls)} pages")
        # Save raw nodes for inspection
        self._save_raw_nodes(nodes)
        return nodes

    def _build_graph(self, nodes: List[DocumentNode]) -> None:
        """Step 2: Build NetworkX graph from scraped nodes."""
        # Add all nodes to graph
        for node in nodes:
            self.graph_store.add_node(node_id=node.node_id, content = node.content,
                                    section_type=node.section_type,
                                    metadata={**node.metadata, 'url': node.url})
            logger.info(f"Added {len(nodes)} nodes to graph")

            #Build edges from hyperlinks
            # Create a URL -> node_id mapping for fast lookup
            url_to_node = {}
            for node in nodes:
                url_to_node[node.url] = node.node_id
            
            edge_count = 0
            for node in nodes:
                for link_url in node.links:
                    # Check if the linked URL was also scrapped
                    if link_url in url_to_node:
                        target_node_id = url_to_node[link_url]
                        self.graph_store.add_edge(
                                                source_id = node.node_id,
                                                target_id = target_node_id,
                                                relationship_type="hyperlink"
                                                )
                        edge_count += 1
            logger.info(f"Added {edge_count} hyperlink edges to graph")

        #Save graph to disk
        self.graph_store.save()
        logger.info(f"Graph saved with {self.graph_store.graph.number_of_nodes()} nodes, {self.graph_store.graph.number_of_edges()} edges")
        
    def _store_vectors(self, nodes: List[DocumentNode]) -> None:
        """Step 3: Store embeddings in ChromaDB."""
        # Deduplicate nodes by node_id (keep first occurrence)
        seen_ids = set()
        unique_nodes = []
        for node in nodes:
            if node.node_id not in seen_ids:
                seen_ids.add(node.node_id)
                unique_nodes.append(node)
        
        if len(unique_nodes) < len(nodes):
            logger.warning(f"Removed {len(nodes) - len(unique_nodes)} duplicate nodes")
        
        # Prepare data for vector store
        documents = [node.content for node in unique_nodes]
        node_ids = [node.node_id for node in unique_nodes]
        metadatas = [{**node.metadata, 'url': node.url, 'section_type': node.section_type} for node in unique_nodes]

        # Add to vector store
        self.vector_store.add_documents(documents=documents, node_ids=node_ids, metadatas=metadatas)

        stats = self.vector_store.get_collection_stats()
        logger.info(f"Vector store contains {stats['total_documents']} documents")
    
    def _save_raw_nodes(self, nodes: List[DocumentNode]) -> None:
        """Save raw scraped nodesas JSON for inspection."""
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_data = [{
                    'node_id': node.node_id,
                    'content': node.content[:200] + "..." if len(node.content) > 200 else node.content,
                    'section_type': node.section_type,
                    'url': node.url,
                    'num_links': len(node.links),
                    'metadata': node.metadata
                    } for node in nodes
                    ]
        output_path = DATA_RAW_DIR / "scraped_nodes.json"
        with open(output_path, 'w') as f:
            json.dump(raw_data, f, indent=2)
        
        logger.info(f"Raw nodes saved to {output_path}")
    
    def _save_metadata(self, base_url: str, nodes: List[DocumentNode]) -> dict:
        """Save pipeline metadata."""
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        graph_stats = self.graph_store.get_stats()
        vector_stats = self.vector_store.get_collection_stats()

        metadata = {
            'source_url': base_url,
            'total_nodes': len(nodes),
            'section_type_counts': {},
            'graph_edges': graph_stats['num_edges'],
            'vector_db_count': vector_stats['total_documents'],
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model
        }

        #count section types
        for node in nodes:
            section_type = node.section_type
            metadata['section_type_counts'][section_type] = metadata['section_type_counts'].get(section_type, 0) + 1
        
        output_path = DATA_PROCESSED_DIR / "ingestion_metadata.json"
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {output_path}")
        return metadata

# Test the pipeline
if __name__ == "__main__":
    print("Testing Ingestion Pipeline...\n")
    print("WARNING: This will scrape real documentation and store data.\n")
    
    # Initialize pipeline
    pipeline = IngestionPipeline(
        collection_name="test_ingestion",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    # Run on small test dataset
    test_url = "https://docs.python.org/3/tutorial/introduction.html"
    
    print(f"Running pipeline on: {test_url}\n")
    
    stats = pipeline.run(
        base_url=test_url,
        max_pages=3,
        delay=2.0
    )
    print("PIPELINE TEST RESULTS")
    print("="*60)
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Section types: {stats['section_type_counts']}")
    print(f"Graph edges: {stats['graph_edges']}")
    print(f"Vector DB docs: {stats['vector_db_count']}")
    
    print("\n Pipeline test completed!")