import sys
import os
import platform
from utils.logger import get_logger

logger = get_logger(__name__)

def get_device():
    """Auto detect the best available GPU backend for llama-cpp-python
    
    Returns:
        tuple: (backend, n_gpu_layers) where:
            - backend: str - llama-cpp device ('metal', 'cuda', 'vulkan', 'cpu')
            - n_gpu_layers: int - layers to offload to GPU (-1 = all, 0 = none)
    """

    try:
        # Check for Apple Silicon (Metal backend)
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            logger.info("Detected Apple Silicon - using Metal backend")
            return "metal", -1
        
        # Check for NVIDIA CUDA (look for nvidia-smi or CUDA env vars)
        if _has_cuda():
            logger.info("Detected NVIDIA CUDA device")
            return "cuda", -1

        # Check for Vulkan support (Intel/AMD GPUs on Linux/Windows)
        if _has_vulkan():
            logger.info("Detected Vulkan-compatible GPU")
            return "vulkan", -1
        
        # Fallback to CPU
        logger.info("No GPU detected, using CPU")
        return "cpu", 0
        
    except Exception as e:
        logger.warning(f"Error detecting device: {e}, falling back to CPU")
        return "cpu", 0

def _has_cuda():
    """Check if NVIDIA CUDA is available"""
    # Check for nvidia-smi command
    if os.system("nvidia-smi > /dev/null 2>&1") == 0:
        return True
    # Check for CUDA environment variables
    if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
        return True
    return False

def _has_vulkan():
    """Check if Vulkan is available (Intel/AMD GPU support)"""
    # Check for vulkaninfo command (Linux/Windows)
    if os.system("vulkaninfo > /dev/null 2>&1") == 0:
        return True
    return False

if __name__ == "__main__":
    backend, n_gpu_layers = get_device()
    print(f"Selected backend: {backend}, GPU layers: {n_gpu_layers}")