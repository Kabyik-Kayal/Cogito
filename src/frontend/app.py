"""
Cogito Web Application - FastAPI Backend

A brutalist terminal-style web interface for the self-correcting RAG system.

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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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
async def get_status():
    """Get system status."""
    try:
        status = {
            "initialized": app_state.is_initialized,
            "collection": app_state.current_collection,
            "document_count": 0,
            "graph_nodes": 0
        }
        
        # Only fetch stats if system is initialized and graph exists
        if app_state.is_initialized and app_state.graph:
            try:
                # Use cached graph's stores instead of creating new instances
                if hasattr(app_state.graph, 'retrieve_node') and app_state.graph.retrieve_node:
                    vs = app_state.graph.retrieve_node.vector_store
                    status["document_count"] = vs.get_collection_stats()["total_documents"]
                if hasattr(app_state.graph, 'graph_augment_node') and app_state.graph.graph_augment_node:
                    gs = app_state.graph.graph_augment_node.graph_store
                    status["graph_nodes"] = gs.get_stats()["num_nodes"]
            except Exception:
                pass  # Silently fail for stats
        
        return status
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {"initialized": False, "collection": "", "document_count": 0, "graph_nodes": 0}


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
    files: List[UploadFile] = File(...),
    collection: str = Form("cogito_docs"),
    chunking_strategy: str = Form("semantic")
):
    """Upload and ingest documents."""
    try:
        from src.ingestion.parser import DocumentParser, ParsedChunk
        from src.ingestion.scraper import DocumentNode
        from src.db.vector_store import VectorStore
        from src.db.graph_store import GraphStore
        
        # Save uploaded files
        upload_dir = DATA_RAW_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for file in files:
            # Generate unique filename
            ext = Path(file.filename).suffix
            unique_name = f"{uuid.uuid4().hex}{ext}"
            file_path = upload_dir / unique_name
            
            # Save file
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            saved_files.append({
                "original": file.filename,
                "saved": str(file_path),
                "size": len(content)
            })
        
        logger.info(f"Saved {len(saved_files)} files")
        
        # Parse files
        parser = DocumentParser(
            chunk_size=512,
            chunking_strategy=chunking_strategy
        )
        
        all_chunks = []
        for file_info in saved_files:
            try:
                chunks = parser.parse_file(file_info["saved"])
                all_chunks.extend(chunks)
                logger.info(f"Parsed {file_info['original']}: {len(chunks)} chunks")
            except Exception as e:
                logger.warning(f"Failed to parse {file_info['original']}: {e}")
        
        if not all_chunks:
            return {"status": "error", "message": "No content could be parsed from files"}
        
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
        
        # Store in vector DB
        vector_store = VectorStore(collection_name=collection)
        
        ids = [node.node_id for node in nodes]
        contents = [node.content for node in nodes]
        metadatas = [{"source": node.url, "type": node.section_type} for node in nodes]
        
        vector_store.add_documents(ids=ids, documents=contents, metadatas=metadatas)
        
        # Build graph
        graph_store = GraphStore()
        for node in nodes:
            graph_store.add_node(
                node_id=node.node_id,
                content=node.content,
                metadata={"source": node.url, "type": node.section_type}
            )
        graph_store.save()
        
        # Update app state
        app_state.current_collection = collection
        
        return {
            "status": "success",
            "files_processed": len(saved_files),
            "chunks_created": len(all_chunks),
            "collection": collection
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest-url")
async def ingest_from_url(request: IngestURLRequest, background_tasks: BackgroundTasks):
    """Ingest documents from a URL."""
    try:
        from src.ingestion.pipeline import IngestionPipeline
        
        job_id = uuid.uuid4().hex
        app_state.ingestion_status[job_id] = {
            "status": "running",
            "url": request.url,
            "progress": 0,
            "message": "Starting ingestion..."
        }
        
        # Run ingestion in background
        async def run_ingestion():
            try:
                pipeline = IngestionPipeline(collection_name=request.collection)
                stats = pipeline.run(
                    base_url=request.url,
                    max_pages=request.max_pages,
                    delay=1.0
                )
                
                app_state.ingestion_status[job_id] = {
                    "status": "completed",
                    "url": request.url,
                    "progress": 100,
                    "stats": stats
                }
                app_state.current_collection = request.collection
                
            except Exception as e:
                app_state.ingestion_status[job_id] = {
                    "status": "failed",
                    "url": request.url,
                    "error": str(e)
                }
        
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


@app.post("/api/query")
async def query(request: QueryRequest):
    """Run a RAG query."""
    try:
        # Initialize if needed
        if not app_state.is_initialized or app_state.current_collection != request.collection:
            from src.graph import CogitoGraph
            app_state.graph = CogitoGraph(collection_name=request.collection)
            app_state.is_initialized = True
            app_state.current_collection = request.collection
        
        # Run query
        logger.info(f"Query: {request.question}")
        response = app_state.graph.query(request.question)
        
        # Build trace
        trace = [
            {"action": "RETRIEVE", "status": "pass", "details": f"Found {response['sources']['vector_docs']} documents"},
            {"action": "GRAPH_AUGMENT", "status": "pass", "details": f"Added {response['sources']['graph_docs']} neighbors"},
            {"action": "GENERATE", "status": "pass", "details": "Initial answer generated"},
        ]
        
        # Add audit/retry steps
        for i in range(response['retry_count'] + 1):
            if i < response['retry_count']:
                trace.append({"action": "AUDIT", "status": "fail", "details": f"Attempt {i+1} failed"})
                trace.append({"action": "REWRITE", "status": "pass", "details": "Query refined"})
                trace.append({"action": "RETRIEVE", "status": "pass", "details": "New documents retrieved"})
                trace.append({"action": "GENERATE", "status": "pass", "details": f"Draft {i+2} created"})
            else:
                status = "pass" if response['audit_status'] == "pass" else "fail"
                trace.append({"action": "AUDIT", "status": status, "details": response.get('audit_reason', 'Verified')[:50]})
        
        return {
            "answer": response["answer"],
            "audit_status": response["audit_status"],
            "audit_reason": response.get("audit_reason", ""),
            "retry_count": response["retry_count"],
            "sources": response["sources"],
            "trace": trace
        }
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run with: uvicorn src.frontend.app:app --reload --port 5000
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
