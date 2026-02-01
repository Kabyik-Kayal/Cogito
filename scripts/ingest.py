"""
Ingestion Script

CLI entry point for running the ingestion pipeline.

Usage:
    # Scrape from URL (default)
    python scripts/ingest.py --url "https://docs.nvidia.com/cuda/"
    
    # Parse local files
    python scripts/ingest.py --local /path/to/docs/ --format md
    
    # Custom settings
    python scripts/ingest.py --url "https://kubernetes.io/docs/" --max-pages 50 --collection my_docs
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.parser import DocumentParser, ParsedChunk
from src.ingestion.scraper import DocumentNode
from src.db.vector_store import VectorStore
from src.db.graph_store import GraphStore
from utils.logger import get_logger

logger = get_logger(__name__)


def run_web_ingestion(
    url: str,
    collection_name: str,
    max_pages: int,
    delay: float,
    embedding_model: str
):
    """Run ingestion from web URL."""
    print("COGITO INGESTION PIPELINE - WEB SCRAPING")
    print(f"Source URL: {url}")
    print(f"Collection: {collection_name}")
    print(f"Max Pages: {max_pages}")

    
    pipeline = IngestionPipeline(
        collection_name=collection_name,
        embedding_model=embedding_model
    )
    
    stats = pipeline.run(
        base_url=url,
        max_pages=max_pages,
        delay=delay
    )

    print("INGESTION COMPLETE")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Section types: {stats['section_type_counts']}")
    print(f"Graph edges: {stats['graph_edges']}")
    print(f"Vector DB docs: {stats['vector_db_count']}")
    
    return stats


def run_local_ingestion(
    path: str,
    collection_name: str,
    file_format: str,
    chunking_strategy: str,
    chunk_size: int,
    embedding_model: str
):
    """Run ingestion from local files."""
    print("COGITO INGESTION PIPELINE - LOCAL FILES")
    print(f"Source Path: {path}")
    print(f"Collection: {collection_name}")
    print(f"Format Filter: {file_format or 'all supported'}")
    print(f"Chunking: {chunking_strategy} (size={chunk_size})")
    
    # Parse documents
    extensions = None
    if file_format:
        ext_map = {
            'md': ['.md', '.markdown'],
            'rst': ['.rst'],
            'html': ['.html', '.htm'],
            'txt': ['.txt']
        }
        extensions = ext_map.get(file_format.lower())
    
    parser = DocumentParser(
        chunk_size=chunk_size,
        chunking_strategy=chunking_strategy
    )
    
    path_obj = Path(path)
    if path_obj.is_file():
        chunks = parser.parse_file(path)
    else:
        chunks = parser.parse_directory(path, extensions=extensions)
    
    if not chunks:
        print("No documents found to ingest!")
        return None
    
    print(f"Parsed {len(chunks)} chunks")
    
    # Convert ParsedChunks to DocumentNodes for compatibility
    nodes = []
    for chunk in chunks:
        node = DocumentNode(
            node_id=chunk.chunk_id,
            content=chunk.content,
            section_type=chunk.chunk_type,
            url=f"file://{chunk.source_file}",
            metadata=chunk.metadata
        )
        nodes.append(node)
    
    # Store in vector DB
    print("Storing embeddings in ChromaDB...")
    vector_store = VectorStore(
        collection_name=collection_name,
        embedding_model=embedding_model
    )
    
    # Add documents
    ids = [node.node_id for node in nodes]
    contents = [node.content for node in nodes]
    metadatas = [{"source": node.url, "type": node.section_type, **node.metadata} for node in nodes]
    
    vector_store.add_documents(ids=ids, documents=contents, metadatas=metadatas)
    
    # Build graph
    print("Building document graph...")
    graph_store = GraphStore()
    
    for node in nodes:
        graph_store.add_node(
            node_id=node.node_id,
            content=node.content,
            metadata={"source": node.url, "type": node.section_type}
        )
    
    # Add edges based on source file (nodes from same file are related)
    from collections import defaultdict
    file_nodes = defaultdict(list)
    for node in nodes:
        file_nodes[node.url].append(node.node_id)
    
    edge_count = 0
    for file_url, node_ids in file_nodes.items():
        for i in range(len(node_ids) - 1):
            graph_store.add_edge(node_ids[i], node_ids[i + 1], edge_type="sequential")
            edge_count += 1
    
    graph_store.save()
    
    stats = {
        "total_chunks": len(chunks),
        "vector_db_count": vector_store.get_collection_stats()['total_documents'],
        "graph_nodes": graph_store.get_stats()['nodes'],
        "graph_edges": edge_count,
        "files_processed": len(file_nodes),
        "chunking_strategy": chunking_strategy
    }

    print("INGESTION COMPLETE")
    print(f"Chunks created: {stats['total_chunks']}")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Vector DB docs: {stats['vector_db_count']}")
    print(f"Graph edges: {stats['graph_edges']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Cogito Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape web documentation
  python scripts/ingest.py --url "https://docs.python.org/3/tutorial/"
  
  # Parse local markdown files
  python scripts/ingest.py --local ./docs/ --format md
  
  # Parse with custom chunking
  python scripts/ingest.py --local ./docs/ --strategy semantic --chunk-size 400
        """
    )
    
    # Source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--url",
        type=str,
        help="URL to scrape documentation from"
    )
    source_group.add_argument(
        "--local",
        type=str,
        help="Local path to file or directory"
    )
    
    # Common options
    parser.add_argument(
        "--collection",
        type=str,
        default="cogito_docs",
        help="ChromaDB collection name (default: cogito_docs)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="HuggingFace embedding model (default: all-MiniLM-L6-v2)"
    )
    
    # Web scraping options
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Maximum pages to scrape (default: 20)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between requests in seconds (default: 2.0)"
    )
    
    # Local parsing options
    parser.add_argument(
        "--format",
        type=str,
        choices=['md', 'rst', 'html', 'txt'],
        help="Filter local files by format"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=['fixed', 'semantic', 'sliding'],
        default='semantic',
        help="Chunking strategy (default: semantic)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in characters (default: 512)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.url:
            stats = run_web_ingestion(
                url=args.url,
                collection_name=args.collection,
                max_pages=args.max_pages,
                delay=args.delay,
                embedding_model=args.embedding_model
            )
        else:
            stats = run_local_ingestion(
                path=args.local,
                collection_name=args.collection,
                file_format=args.format,
                chunking_strategy=args.strategy,
                chunk_size=args.chunk_size,
                embedding_model=args.embedding_model
            )
        
        if stats:
            print("Ingestion successful!")
            return 0
        else:
            print("Ingestion failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
