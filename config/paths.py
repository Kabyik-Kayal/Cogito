import os
import pathlib


# Base directories
BASE_DIR = pathlib.Path(__file__).parent.parent

# Model directories
MODEL_DIR = BASE_DIR / "models"
ONNX_CACHE_DIR = MODEL_DIR / "onnx_cache"  # Persistent cache for ONNX embedding models
MISTRAL_GGUF_MODEL_PATH = MODEL_DIR / "models--mistralai--Ministral-3-8B-Reasoning-2512-GGUF/snapshots/4413bc9b7b091ddad28101650d1395f5e045a328/Ministral-3-8B-Reasoning-2512-Q8_0.gguf"

# Data directories (for scraped docs)
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Database directories
DB_DIR = BASE_DIR / "db"
CHROMA_DB_DIR = DB_DIR / "chroma"
GRAPH_STORE_DIR = DB_DIR / "graph"

# Graph files
GRAPH_PICKLE_PATH = GRAPH_STORE_DIR / "doc_graph.pkl"
GRAPH_METADATA_PATH = GRAPH_STORE_DIR / "graph_metadata.json"

# Logs directory
LOGS_DIR = BASE_DIR / "logs"

# Results directory (for evaluation)
RESULTS_DIR = BASE_DIR / "results"
