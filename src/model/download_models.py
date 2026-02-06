from huggingface_hub import hf_hub_download, hf_hub_url
from config.paths import *
from utils.logger import get_logger
from utils.custom_exception import CustomException
import os
import requests
from tqdm import tqdm
import tarfile

logger = get_logger(__name__)

def download_model_with_progress(url, destination, progress_callback=None):
    """Download a file with progress tracking."""
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192  # 8KB chunks
    downloaded = 0
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                if progress_callback and total_size > 0:
                    percent = int((downloaded / total_size) * 90) + 10  # 10-100% range
                    progress_callback(percent, f"Downloaded {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB")

def download_onnx_model(progress_callback=None):
    """
    Download ChromaDB ONNX embedding model by initializing a test collection.
    This lets ChromaDB handle the download to its expected location automatically.
    Uses symlink to persist the model in Docker volume at /app/models/onnx_cache.
    """
    try:
        from pathlib import Path
        import chromadb
        from chromadb.config import Settings
        from chromadb.utils import embedding_functions
        import shutil
        
        # Set up symlink so ChromaDB's cache points to our Docker volume
        # ChromaDB expects: ~/.cache/chroma/onnx_models
        # We persist at: /app/models/onnx_cache/chroma (which is in Docker volume)
        home_cache = Path.home() / ".cache" / "chroma"
        onnx_models_link = home_cache / "onnx_models"
        persistent_onnx_dir = ONNX_CACHE_DIR / "chroma"
        
        # Create the persistent directory structure
        persistent_onnx_dir.mkdir(parents=True, exist_ok=True)
        
        # If the link doesn't exist or points to the wrong place, set it up
        if not onnx_models_link.exists() or not onnx_models_link.is_symlink():
            # Remove if it's a regular directory
            if onnx_models_link.exists() and not onnx_models_link.is_symlink():
                shutil.rmtree(onnx_models_link)
            # Create parent directory
            home_cache.mkdir(parents=True, exist_ok=True)
            # Create symlink
            onnx_models_link.symlink_to(persistent_onnx_dir)
            logger.info(f"Created symlink: {onnx_models_link} -> {persistent_onnx_dir}")
        
        # Check if model exists (now in persistent location via symlink)
        chroma_model_file = persistent_onnx_dir / "all-MiniLM-L6-v2" / "onnx" / "model.onnx"
        
        if chroma_model_file.exists():
            logger.info("ONNX embedding model already exists in persistent storage")
            if progress_callback:
                progress_callback(10, "Embedding model ready")
            return
        
        logger.info("Initializing ONNX embedding model download...")
        if progress_callback:
            progress_callback(2, "Downloading embedding model (~80MB)...")
        
        # Create a temporary ChromaDB client
        temp_db_path = ONNX_CACHE_DIR / "temp_chroma"
        temp_db_path.mkdir(parents=True, exist_ok=True)
        
        client = chromadb.PersistentClient(
            path=str(temp_db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create embedding function (this triggers the download to persistent location via symlink)
        embedding_function = embedding_functions.ONNXMiniLM_L6_V2()
        
        if progress_callback:
            progress_callback(5, "Initializing embedding model...")
        
        # Create a test collection with a sample document
        collection = client.get_or_create_collection(
            name="test_onnx_download",
            embedding_function=embedding_function
        )
        
        # Add a test document (this ensures the model is fully loaded)
        collection.add(
            documents=["Test document to trigger ONNX model download"],
            ids=["test_id"]
        )
        
        if progress_callback:
            progress_callback(8, "Embedding model downloaded successfully")
        
        # Clean up: delete the test collection and temporary database
        client.delete_collection("test_onnx_download")
        
        # Remove temporary database directory
        if temp_db_path.exists():
            shutil.rmtree(temp_db_path)
        
        logger.info(f"ONNX embedding model downloaded to persistent storage: {chroma_model_file}")
        if progress_callback:
            progress_callback(10, "Embedding model ready")
            
    except Exception as e:
        logger.error(f"Failed to download ONNX model: {e}")
        # Don't fail the entire download if ONNX fails
        if progress_callback:
            progress_callback(10, "Embedding model will download on first use")

def download_model(progress_callback=None):
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created model directory: {MODEL_DIR}")
    
    try:
        # Download ONNX embedding model first (smaller, faster)
        download_onnx_model(progress_callback)
        
        if not MISTRAL_GGUF_MODEL_PATH.exists():
            logger.info("Downloading Ministral 3 8B Reasoning 8-bit Quantized GGUF model")
            
            if progress_callback:
                progress_callback(5, "Fetching download URL...")
            
            # Get the download URL from HuggingFace
            repo_id = "mistralai/Ministral-3-8B-Reasoning-2512-GGUF"
            filename = "Ministral-3-8B-Reasoning-2512-Q8_0.gguf"
            
            try:
                # Try to get URL for downloading
                url = hf_hub_url(repo_id=repo_id, filename=filename)
                
                if progress_callback:
                    progress_callback(10, "Starting download (this may take several minutes)...")
                
                # Download with progress tracking
                temp_path = str(MISTRAL_GGUF_MODEL_PATH) + ".tmp"
                download_model_with_progress(url, temp_path, progress_callback)
                
                # Move temp file to final location
                os.rename(temp_path, MISTRAL_GGUF_MODEL_PATH)
                
            except Exception as e:
                logger.warning(f"Custom download failed, falling back to hf_hub_download: {e}")
                # Fallback to standard download
                if progress_callback:
                    progress_callback(10, "Downloading (progress unavailable)...")
                
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=MODEL_DIR,
                    resume_download=True
                )
            
            if progress_callback:
                progress_callback(100, "Download complete")
            
            logger.info(f"Model downloaded to: {MISTRAL_GGUF_MODEL_PATH}")
        else:
            logger.info("Model already exists.")
            if progress_callback:
                progress_callback(100, "Model already exists")

    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        if progress_callback:
            progress_callback(-1, f"Error: {str(e)}")
        raise CustomException(e, "Error downloading model")

if __name__ == "__main__":
    download_model()