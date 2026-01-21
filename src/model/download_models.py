from huggingface_hub import hf_hub_download
from config.paths import *
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

def download_model():
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created model directory: {MODEL_DIR}")
    
    try:
        if not MISTRAL_GGUF_MODEL_PATH.exists():
            logger.info("Downloading Ministral 3 8B Reasoning 8-bit Quantized GGUF model")
            #Download Ministral 3 8B Reasoning 8-bit Quantized GGUF model
            model_path = hf_hub_download(
                repo_id="mistralai/Ministral-3-8B-Reasoning-2512-GGUF",
                filename="Ministral-3-8B-Reasoning-2512-Q8_0.gguf",
                cache_dir=MODEL_DIR)
            logger.info(f"Model downloaded to: {model_path}")
        else:
            logger.info("Model already exists.")

    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise CustomException(e, "Error downloading model")

if __name__ == "__main__":
    download_model()