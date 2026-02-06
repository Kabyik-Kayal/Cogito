"""
Cogito Web Application - FastAPI Backend

A terminal-style web interface for the self-correcting RAG system.

Features:
- File upload (PDF, MD, TXT, HTML)
- URL-based ingestion
- Full RAG query interface with trace display

Run: uvicorn src.frontend.app:app --reload --port 5000
"""

import os
import sys
import uuid
import asyncio
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request
from pydantic import BaseModel

from config.paths import DATA_RAW_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="Cogito",
    description="Self-Correcting RAG System",
    version="1.0.0"
)

# Static files and templates
frontend_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=frontend_dir / "static"), name="static")
templates = Jinja2Templates(directory=frontend_dir / "templates")

# ============================================================================
# Global State (for demo - use Redis/DB in production)
# ============================================================================

class AppState:
    def __init__(self):
        self.graph = None
        self.is_initialized = False
        self.current_collection = "cogito_docs"
        self.ingestion_status = {}
        self.query_jobs = {} # Store query status by job_id
        
app_state = AppState()

# ============================================================================
# Request/Response Models
# ============================================================================

class QueryRequest(BaseModel):
    question: str
    collection: Optional[str] = "cogito_docs"

class QueryResponse(BaseModel):
    answer: str
    audit_status: str
    audit_reason: str
    retry_count: int
    sources: dict
    trace: list

class IngestURLRequest(BaseModel):
    url: str
    collection: Optional[str] = "cogito_docs"
    max_pages: Optional[int] = 20

class StatusResponse(BaseModel):
    initialized: bool
    collection: str
    document_count: int
    graph_nodes: int

# ============================================================================
# Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main application page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def get_status(collection: str = None):
    """Get system status for specified collection or current collection."""
    try:
        from src.db.vector_store import VectorStore
        from src.db.graph_store import GraphStore
        
        target_collection = collection or app_state.current_collection
        
        status = {
            "initialized": app_state.is_initialized,
            "collection": target_collection,
            "document_count": 0,
            "graph_nodes": 0,
            "graph_edges": 0
        }
        
        # Query stores directly to get accurate counts
        try:
            vs = VectorStore(collection_name=target_collection)
            stats = vs.get_collection_stats()
            status["document_count"] = stats["total_documents"]
        except Exception as e:
            logger.debug(f"Could not get vector stats: {e}")
        
        try:
            gs = GraphStore(collection_name=target_collection)
            graph_stats = gs.get_stats()
            status["graph_nodes"] = graph_stats["num_nodes"]
            status["graph_edges"] = graph_stats["num_edges"]
        except Exception as e:
            logger.debug(f"Could not get graph stats: {e}")
        
        return status
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {"initialized": False, "collection": "", "document_count": 0, "graph_nodes": 0, "graph_edges": 0}


@app.get("/api/collections")
async def list_collections():
    """List all available collections in ChromaDB."""
    try:
        import chromadb
        from chromadb.config import Settings
        from config.paths import CHROMA_DB_DIR
        
        client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        collections = client.list_collections()
        return {
            "collections": [{"name": c.name, "count": c.count()} for c in collections],
            "current": app_state.current_collection
        }
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        return {"collections": [], "current": app_state.current_collection}


@app.post("/api/switch-collection")
async def switch_collection(collection: str):
    """Switch to a different collection."""
    try:
        app_state.current_collection = collection
        app_state.is_initialized = False  # Reset initialization
        app_state.graph = None
        logger.info(f"Switched to collection: {collection}")
        return {"status": "success", "collection": collection}
    except Exception as e:
        logger.error(f"Failed to switch collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.delete("/api/delete-collection")
async def delete_collection(collection: str):
    """
    Permanently delete a collection including all vectors and graph data.
    
    Args:
        collection: Name of the collection to delete
    """
    try:
        from src.db.vector_store import VectorStore
        from src.db.graph_store import GraphStore
        
        deleted = []
        
        # Delete from ChromaDB using VectorStore (avoids client conflict)
        try:
            vs = VectorStore(collection_name=collection)
            vs.delete_collection()
            deleted.append("vector collection")
            logger.info(f"Deleted ChromaDB collection: {collection}")
        except Exception as e:
            logger.warning(f"Could not delete vector collection: {e}")
        
        # Delete graph using GraphStore
        try:
            gs = GraphStore(collection_name=collection, auto_load=False)
            gs.delete_graph()
            deleted.append("graph data")
        except Exception as e:
            logger.warning(f"Could not delete graph: {e}")
        
        # Reset app state if current collection was deleted
        if collection == app_state.current_collection:
            app_state.is_initialized = False
            app_state.current_collection = "cogito_docs"
            app_state.graph = None
        
        return {"status": "success", "collection": collection, "deleted": deleted}
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/initialize")
async def initialize_system(collection: str = "cogito_docs"):
    """Initialize the RAG system."""
    try:
        logger.info(f"Initializing Cogito with collection: {collection}")
        
        from src.graph import CogitoGraph
        
        app_state.graph = CogitoGraph(collection_name=collection)
        app_state.is_initialized = True
        app_state.current_collection = collection
        
        return {"status": "success", "message": "System initialized", "collection": collection}
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    collection: str = Form("cogito_docs"),
    chunking_strategy: str = Form("semantic")
):
    """Upload and ingest documents (async with progress tracking)."""
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save files first (before background task)
        upload_dir = DATA_RAW_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for file in files:
            ext = Path(file.filename).suffix
            unique_name = f"{uuid.uuid4().hex}{ext}"
            file_path = upload_dir / unique_name
            
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            saved_files.append({
                "original": file.filename,
                "saved": str(file_path),
                "size": len(content)
            })
        
        # Initialize job status
        app_state.ingestion_status[job_id] = {
            "status": "running",
            "step": "upload",
            "progress": 10,
            "files_processed": len(saved_files),
            "chunks_created": 0,
            "nodes_created": 0,
            "message": f"Uploaded {len(saved_files)} files",
            "activity_log": [f"[UPLOAD] Received {len(saved_files)} files"],
            "error": None
        }
        
        # Start background processing
        background_tasks.add_task(
            process_uploaded_files,
            job_id, saved_files, collection, chunking_strategy
        )
        
        return {"status": "started", "job_id": job_id}
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def process_uploaded_files(job_id: str, saved_files: list, collection: str, chunking_strategy: str):
    """Background task to process uploaded files."""
    try:
        from src.ingestion.parser import DocumentParser
        from src.ingestion.scraper import DocumentNode
        from src.db.vector_store import VectorStore
        from src.db.graph_store import GraphStore
        
        status = app_state.ingestion_status[job_id]
        
        # Step 2: Parse and chunk files
        status["step"] = "chunk"
        status["progress"] = 20
        status["message"] = "Parsing and chunking files..."
        status["activity_log"].append("[CHUNK] Starting document parsing...")
        
        parser = DocumentParser(
            chunk_size=512,
            chunking_strategy=chunking_strategy
        )
        
        all_chunks = []
        for i, file_info in enumerate(saved_files):
            try:
                chunks = parser.parse_file(file_info["saved"])
                all_chunks.extend(chunks)
                status["activity_log"].append(f"[CHUNK] {file_info['original']}: {len(chunks)} chunks")
                status["chunks_created"] = len(all_chunks)
                status["progress"] = 20 + int((i + 1) / len(saved_files) * 20)
            except Exception as e:
                status["activity_log"].append(f"[WARN] Failed to parse {file_info['original']}: {e}")
        
        if not all_chunks:
            status["status"] = "failed"
            status["error"] = "No content could be parsed from files"
            return
        
        # Convert to DocumentNodes
        nodes = []
        for chunk in all_chunks:
            node = DocumentNode(
                node_id=chunk.chunk_id,
                content=chunk.content,
                section_type=chunk.chunk_type,
                url=f"file://{chunk.source_file}",
                metadata=chunk.metadata
            )
            nodes.append(node)
        
        # Step 3: Build graph
        status["step"] = "graph"
        status["progress"] = 50
        status["message"] = "Building knowledge graph..."
        status["activity_log"].append(f"[GRAPH] Processing {len(nodes)} nodes...")
        
        graph_store = GraphStore(collection_name=collection, auto_load=True)
        existing_count = graph_store.get_stats()["num_nodes"]
        
        nodes_added = 0
        for i, node in enumerate(nodes):
            if node.node_id not in graph_store.graph:
                graph_store.add_node(
                    node_id=node.node_id,
                    content=node.content,
                    metadata={"source": node.url, "type": node.section_type}
                )
                nodes_added += 1
            status["nodes_created"] = nodes_added
            status["progress"] = 50 + int((i + 1) / len(nodes) * 20)
        
        graph_store.save()
        status["activity_log"].append(f"[GRAPH] Added {nodes_added} nodes (existing: {existing_count})")
        
        # Step 4: Store in vector DB
        status["step"] = "vectors"
        status["progress"] = 75
        status["message"] = "Creating vector embeddings..."
        status["activity_log"].append("[VECTORS] Initializing vector store...")
        
        vector_store = VectorStore(collection_name=collection)
        
        node_ids = [node.node_id for node in nodes]
        contents = [node.content for node in nodes]
        metadatas = [{"source": node.url, "type": node.section_type} for node in nodes]
        
        vector_store.add_documents(node_ids=node_ids, documents=contents, metadatas=metadatas)
        
        status["activity_log"].append(f"[VECTORS] Stored {len(nodes)} embeddings")
        
        # Complete
        status["step"] = "done"
        status["progress"] = 100
        status["status"] = "completed"
        status["message"] = "Upload complete!"
        status["activity_log"].append(f"[DONE] Ingestion complete: {len(all_chunks)} chunks, {nodes_added} graph nodes")
        
        # Update app state
        app_state.current_collection = collection
        
    except Exception as e:
        logger.error(f"Upload processing failed: {e}")
        status = app_state.ingestion_status.get(job_id)
        if status:
            status["status"] = "failed"
            status["error"] = str(e)
            status["activity_log"].append(f"[ERROR] {str(e)}")


@app.get("/api/upload-status/{job_id}")
async def get_upload_status(job_id: str):
    """Get the status of an upload job."""
    status = app_state.ingestion_status.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@app.post("/api/ingest-url")
async def ingest_from_url(request: IngestURLRequest, background_tasks: BackgroundTasks):
    """Ingest documents from a URL."""
    try:
        from src.ingestion.pipeline import IngestionPipeline
        from src.ingestion.scraper import DocumentationScraper
        from src.db.graph_store import GraphStore
        from src.db.vector_store import VectorStore
        
        job_id = uuid.uuid4().hex
        start_time = datetime.now()
        
        def add_activity(msg: str):
            """Add timestamped activity to log."""
            timestamp = datetime.now().strftime("%H:%M:%S")
            if "activity_log" not in app_state.ingestion_status[job_id]:
                app_state.ingestion_status[job_id]["activity_log"] = []
            app_state.ingestion_status[job_id]["activity_log"].append(f"[{timestamp}] {msg}")
            # Keep last 50 entries
            app_state.ingestion_status[job_id]["activity_log"] = app_state.ingestion_status[job_id]["activity_log"][-50:]
        
        app_state.ingestion_status[job_id] = {
            "status": "running",
            "url": request.url,
            "progress": 0,
            "current_step": "initializing",
            "step_number": 0,
            "total_steps": 4,
            "pages_scraped": 0,
            "total_pages": request.max_pages,
            "nodes_created": 0,
            "message": "Initializing pipeline...",
            "activity_log": []
        }
        add_activity(f"Starting ingestion from {request.url}")
        add_activity(f"Target: up to {request.max_pages} pages")
        
        # Run ingestion in background with progress tracking
        async def run_ingestion():
            try:
                # Step 1: SCRAPING
                app_state.ingestion_status[job_id].update({
                    "current_step": "scraping",
                    "step_number": 1,
                    "progress": 10,
                    "message": "Scraping documentation..."
                })
                add_activity("Phase 1/4: SCRAPING started")
                
                scraper = DocumentationScraper(
                    base_url=request.url,
                    max_pages=request.max_pages,
                    delay=1.0
                )
                
                # Scrape with progress updates
                nodes = []
                scraper._init_scrape()
                
                while scraper.url_queue and len(scraper.visited_urls) < scraper.max_pages:
                    current_url = scraper.url_queue.popleft()
                    if current_url in scraper.visited_urls:
                        continue
                    
                    page_nodes = scraper._scrape_page(current_url)
                    if page_nodes:
                        nodes.extend(page_nodes)
                    
                    # Update progress
                    pages_done = len(scraper.visited_urls)
                    app_state.ingestion_status[job_id].update({
                        "pages_scraped": pages_done,
                        "nodes_created": len(nodes),
                        "progress": 10 + int((pages_done / request.max_pages) * 40),
                        "message": f"Scraping page {pages_done}/{request.max_pages}..."
                    })
                    add_activity(f"Scraped: {current_url[:60]}{'...' if len(current_url) > 60 else ''}")
                    
                    await asyncio.sleep(0.1)  # Allow status updates to propagate
                
                add_activity(f"Scraping complete: {len(nodes)} nodes from {len(scraper.visited_urls)} pages")
                
                if not nodes:
                    raise Exception("No content could be scraped from the URL")
                
                # Step 2: BUILDING GRAPH
                app_state.ingestion_status[job_id].update({
                    "current_step": "graph",
                    "step_number": 2,
                    "progress": 55,
                    "message": "Building knowledge graph..."
                })
                add_activity("Phase 2/4: GRAPH CONSTRUCTION started")
                await asyncio.sleep(0.1)  # Allow UI to update
                
                # Load existing graph to append (auto_load=True loads from disk if exists)
                graph_store = GraphStore(collection_name=request.collection, auto_load=True)
                existing_stats = graph_store.get_stats()
                
                if existing_stats["num_nodes"] > 0:
                    add_activity(f"Loaded existing graph: {existing_stats['num_nodes']} nodes, {existing_stats['num_edges']} edges")
                    add_activity("APPENDING new nodes to existing graph...")
                else:
                    add_activity("Creating new graph (no existing data found)")
                
                await asyncio.sleep(0.1)
                
                # Add nodes to graph with progress updates
                total_nodes = len(nodes)
                nodes_added = 0
                nodes_skipped = 0
                
                for i, node in enumerate(nodes):
                    # Check if node already exists
                    if node.node_id not in graph_store.graph:
                        graph_store.add_node(
                            node_id=node.node_id,
                            content=node.content,
                            section_type=node.section_type,
                            metadata={**node.metadata, 'url': node.url}
                        )
                        nodes_added += 1
                    else:
                        nodes_skipped += 1
                    
                    if i % 10 == 0:
                        progress = 55 + int((i / total_nodes) * 10)  # 55-65% for graph nodes
                        app_state.ingestion_status[job_id].update({
                            "progress": progress,
                            "message": f"Processing nodes ({i+1}/{total_nodes})..."
                        })
                        add_activity(f"Processed {i+1}/{total_nodes} nodes (added: {nodes_added}, existing: {nodes_skipped})")
                        await asyncio.sleep(0.05)  # Allow UI to update
                
                add_activity(f"Graph update complete: {nodes_added} new nodes, {nodes_skipped} already existed")
                await asyncio.sleep(0.1)
                
                # Build edges from hyperlinks
                app_state.ingestion_status[job_id].update({
                    "progress": 65,
                    "message": "Building graph edges..."
                })
                add_activity("Building hyperlink edges...")
                await asyncio.sleep(0.1)
                
                url_to_node = {node.url: node.node_id for node in nodes}
                edge_count = 0
                for node in nodes:
                    for link_url in node.links:
                        if link_url in url_to_node:
                            target_node_id = url_to_node[link_url]
                            graph_store.add_edge(
                                source_id=node.node_id,
                                target_id=target_node_id,
                                relationship_type="hyperlink"
                            )
                            edge_count += 1
                
                graph_store.save()
                add_activity(f"Graph saved: {total_nodes} nodes, {edge_count} edges")
                
                app_state.ingestion_status[job_id].update({
                    "progress": 70
                })
                await asyncio.sleep(0.1)
                
                # Step 3: STORING VECTORS
                app_state.ingestion_status[job_id].update({
                    "current_step": "vectors",
                    "step_number": 3,
                    "progress": 72,
                    "message": "Initializing vector store..."
                })
                add_activity("Phase 3/4: VECTOR STORAGE started")
                await asyncio.sleep(0.1)
                
                add_activity("Loading embedding model...")
                app_state.ingestion_status[job_id].update({
                    "progress": 74,
                    "message": "Loading embedding model..."
                })
                await asyncio.sleep(0.1)
                
                vector_store = VectorStore(collection_name=request.collection)
                
                add_activity("Embedding model loaded")
                await asyncio.sleep(0.1)
                
                # Deduplicate nodes
                seen_ids = set()
                unique_nodes = []
                for node in nodes:
                    if node.node_id not in seen_ids:
                        seen_ids.add(node.node_id)
                        unique_nodes.append(node)
                
                documents = [node.content for node in unique_nodes]
                node_ids = [node.node_id for node in unique_nodes]
                metadatas = [{**node.metadata, 'url': node.url, 'section_type': node.section_type} for node in unique_nodes]
                
                app_state.ingestion_status[job_id].update({
                    "progress": 76,
                    "message": f"Computing embeddings for {len(unique_nodes)} documents..."
                })
                add_activity(f"Computing embeddings for {len(unique_nodes)} documents...")
                await asyncio.sleep(0.1)
                
                vector_store.add_documents(documents=documents, node_ids=node_ids, metadatas=metadatas)
                
                app_state.ingestion_status[job_id].update({
                    "progress": 88,
                    "message": "Embeddings stored successfully"
                })
                add_activity(f"Stored {len(unique_nodes)} documents in vector DB")
                await asyncio.sleep(0.1)
                
                app_state.ingestion_status[job_id].update({
                    "progress": 90
                })
                await asyncio.sleep(0.1)
                
                # Step 4: FINALIZING
                app_state.ingestion_status[job_id].update({
                    "current_step": "finalizing",
                    "step_number": 4,
                    "progress": 95,
                    "message": "Finalizing..."
                })
                add_activity("Phase 4/4: FINALIZING")
                
                # Build stats
                stats = {
                    "total_nodes": len(nodes),
                    "pages_visited": len(scraper.visited_urls),
                    "graph_edges": edge_count,
                    "vector_db_count": len(unique_nodes),
                    "collection_name": request.collection
                }
                
                add_activity("=" * 40)
                add_activity("INGESTION COMPLETE")
                add_activity(f"Pages scraped: {stats['pages_visited']}")
                add_activity(f"Nodes created: {stats['total_nodes']}")
                add_activity(f"Graph edges: {stats['graph_edges']}")
                add_activity(f"Vector docs: {stats['vector_db_count']}")
                
                app_state.ingestion_status[job_id].update({
                    "status": "completed",
                    "current_step": "complete",
                    "step_number": 4,
                    "progress": 100,
                    "message": "Ingestion complete!",
                    "stats": stats
                })
                app_state.current_collection = request.collection
                
            except Exception as e:
                add_activity(f"ERROR: {str(e)}")
                app_state.ingestion_status[job_id].update({
                    "status": "failed",
                    "current_step": "error",
                    "message": f"Failed: {str(e)}",
                    "error": str(e)
                })
        
        background_tasks.add_task(run_ingestion)
        
        return {"status": "started", "job_id": job_id}
        
    except Exception as e:
        logger.error(f"URL ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ingestion-status/{job_id}")
async def get_ingestion_status(job_id: str):
    """Get status of an ingestion job."""
    if job_id not in app_state.ingestion_status:
        raise HTTPException(status_code=404, detail="Job not found")
    return app_state.ingestion_status[job_id]


def process_query(job_id: str, question: str, collection: str):
    """Background task to run query streaming."""
    try:
        app_state.query_jobs[job_id]["status"] = "running"
        app_state.query_jobs[job_id]["step"] = "RETRIEVE"
        app_state.query_jobs[job_id]["logs"].append(f"Starting query: {question}")
        
        # Initialize graph if needed
        if not app_state.is_initialized or app_state.current_collection != collection:
            app_state.query_jobs[job_id]["logs"].append(f"Initializing graph for collection: {collection}...")
            from src.graph import CogitoGraph
            app_state.graph = CogitoGraph(collection_name=collection)
            app_state.is_initialized = True
            app_state.current_collection = collection
        
        app_state.query_jobs[job_id]["logs"].append("Graph initialized. Retrieving documents...")
        
        # Stream execution
        final_state = None
        stream = app_state.graph.run_with_updates(question)
        
        for event in stream:
            for node_name, state in event.items():
                app_state.query_jobs[job_id]["trace"].append({"node": node_name, "state": str(state)[:200] + "..."})
                
                if node_name == "retrieve":
                    count = len(state.get("documents", []))
                    app_state.query_jobs[job_id]["logs"].append(f"Retrieved {count} documents.")
                    app_state.query_jobs[job_id]["step"] = "GRAPH"
                    
                elif node_name == "graph_augment":
                    count = len(state.get("graph_augmented_docs", []))
                    app_state.query_jobs[job_id]["logs"].append(f"Found {count} graph neighbors.")
                    app_state.query_jobs[job_id]["step"] = "GENERATE"
                    
                elif node_name == "generate":
                    app_state.query_jobs[job_id]["logs"].append("Draft generated. Auditing...")
                    app_state.query_jobs[job_id]["step"] = "AUDIT"
                    
                elif node_name == "audit":
                    status = state.get("audit_status", "unknown")
                    app_state.query_jobs[job_id]["logs"].append(f"Audit Status: {status}")
                    if status == "pass":
                        app_state.query_jobs[job_id]["step"] = "DONE"
                    else:
                        app_state.query_jobs[job_id]["logs"].append("Audit failed. Rewriting query...")
                        app_state.query_jobs[job_id]["step"] = "RETRIEVE"
                
                elif node_name == "rewrite":
                    new_q = state.get("question", "")
                    app_state.query_jobs[job_id]["logs"].append(f"Rewritten query: {new_q}")
                
                final_state = state

        # Construct final response
        if not final_state:
            raise Exception("No state returned from graph execution")

        response_payload = {
            "answer": final_state.get("final_answer") or final_state.get("generation", "No answer generated."),
            "audit_status": final_state.get("audit_status", "unknown"),
            "audit_reason": final_state.get("audit_reason", ""),
            "retry_count": final_state.get("retry_count", 0),
            "sources": {
                "vector_docs": len(final_state.get("documents", [])),
                "graph_docs": len(final_state.get("graph_augmented_docs", []))
            },
            "trace": app_state.query_jobs[job_id]["trace"]
        }
        
        app_state.query_jobs[job_id]["response"] = response_payload
        app_state.query_jobs[job_id]["status"] = "completed"
        app_state.query_jobs[job_id]["step"] = "DONE"
        app_state.query_jobs[job_id]["logs"].append("Request completed successfully.")

    except Exception as e:
        logger.error(f"Query job failed: {e}")
        app_state.query_jobs[job_id]["status"] = "failed"
        app_state.query_jobs[job_id]["error"] = str(e)
        app_state.query_jobs[job_id]["logs"].append(f"Error: {str(e)}")


@app.post("/api/query")
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    import uuid
    job_id = str(uuid.uuid4())
    
    app_state.query_jobs[job_id] = {
        "status": "pending",
        "step": "START",
        "logs": [],
        "trace": [],
        "response": None,
        "created_at": str(datetime.now())
    }
    
    background_tasks.add_task(process_query, job_id, request.question, request.collection)
    
    return {"job_id": job_id, "status": "started"}

@app.get("/api/query-status/{job_id}")
async def query_status(job_id: str):
    job = app_state.query_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/download-model")
async def download_model_endpoint(background_tasks: BackgroundTasks):
    """Download the LLM and ONNX embedding models if not already present."""
    try:
        from src.model.download_models import download_model
        from config.paths import MISTRAL_GGUF_MODEL_PATH, ONNX_MODEL_PATH
        from pathlib import Path
        
        # Check if both models already exist
        llm_exists = MISTRAL_GGUF_MODEL_PATH.exists()
        
        # Check both our cache and ChromaDB's hardcoded cache location
        chroma_onnx = Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2" / "onnx" / "model.onnx"
        onnx_exists = ONNX_MODEL_PATH.exists() or chroma_onnx.exists()
        
        if llm_exists and onnx_exists:
            return {"status": "exists", "message": "Models already downloaded"}
        
        # Start download in background (will download missing models)
        job_id = str(uuid.uuid4())
        app_state.ingestion_status[job_id] = {
            "status": "running",
            "message": "Starting download...",
            "progress": 0
        }
        
        background_tasks.add_task(run_model_download, job_id)
        return {"status": "started", "job_id": job_id}
        
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_model_download(job_id: str):
    """Background task to download the model with progress tracking."""
    try:
        from src.model.download_models import download_model
        
        status = app_state.ingestion_status[job_id]
        
        def progress_callback(percent, message):
            status["progress"] = percent
            status["message"] = message
            if percent < 0:
                status["status"] = "failed"
                status["error"] = message
        
        # Simulate progress updates since hf_hub_download doesn't provide granular progress
        status["progress"] = 10
        status["message"] = "Downloading from HuggingFace..."
        
        download_model(progress_callback=progress_callback)
        
        status["status"] = "completed"
        status["message"] = "Model downloaded successfully"
        status["progress"] = 100
        
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        status = app_state.ingestion_status.get(job_id)
        if status:
            status["status"] = "failed"
            status["error"] = str(e)
            status["progress"] = -1


@app.get("/api/model-download-status/{job_id}")
async def get_model_download_status(job_id: str):
    """Get the status of a model download job."""
    status = app_state.ingestion_status.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@app.get("/api/model-status")
async def get_model_status():
    """Check if both LLM and ONNX embedding models are downloaded."""
    try:
        from config.paths import MISTRAL_GGUF_MODEL_PATH, ONNX_MODEL_PATH
        from pathlib import Path
        
        llm_exists = MISTRAL_GGUF_MODEL_PATH.exists()
        
        # Check both our cache and ChromaDB's hardcoded cache location
        chroma_onnx = Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2" / "onnx" / "model.onnx"
        onnx_exists = ONNX_MODEL_PATH.exists() or chroma_onnx.exists()
        
        both_downloaded = llm_exists and onnx_exists
        
        return {
            "downloaded": both_downloaded,
            "llm_downloaded": llm_exists,
            "onnx_downloaded": onnx_exists
        }
    except Exception as e:
        return {"downloaded": False, "error": str(e)}


# ============================================================================
# Run with: uvicorn src.frontend.app:app --reload --port 5000
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
