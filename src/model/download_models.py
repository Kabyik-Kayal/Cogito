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
    """Download ChromaDB ONNX embedding model for offline use."""
    try:
        ONNX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Check if model already exists (ChromaDB extracts to onnx_cache/onnx/)
        onnx_model_file = ONNX_CACHE_DIR / "onnx" / "model.onnx"
        
        if onnx_model_file.exists():
            logger.info("ONNX embedding model already exists")
            if progress_callback:
                progress_callback(10, "Embedding model ready")
            return
        
        logger.info("Downloading ONNX embedding model...")
        if progress_callback:
            progress_callback(2, "Downloading embedding model (~80MB)...")
        
        # Download from ChromaDB's S3 bucket
        onnx_url = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
        temp_tar = ONNX_CACHE_DIR / "onnx_temp.tar.gz"
        
        response = requests.get(onnx_url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(temp_tar, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size > 0:
                        percent = int((downloaded / total_size) * 5) + 2  # 2-7% range
                        progress_callback(percent, f"Downloading embedding model... {downloaded // (1024*1024)}MB")
        
        if progress_callback:
            progress_callback(7, "Extracting embedding model...")
        
        # Extract the tar.gz
        with tarfile.open(temp_tar, 'r:gz') as tar:
            tar.extractall(path=ONNX_CACHE_DIR)
        
        # Clean up temp file
        temp_tar.unlink()
        
        logger.info(f"ONNX embedding model downloaded to: {onnx_model_path}")
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