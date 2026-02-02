import sys
import torch
from utils.logger import get_logger

logger = get_logger(__name__)

def get_device():
    """Auto detect the best available GPU backend
    
    Returns:
        tuple: (device, gpu) where:
            - device: str - PyTorch device name ('mps', 'cuda', 'xpu', 'cpu')
            - gpu: int - -1 for GPU (use whole GPU), 0 for CPU
    """

    try:
        # Check for Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Detected Apple Silicon MPS device")
            return "mps", -1
        
        # Check for NVIDIA CUDA GPU
        elif torch.cuda.is_available():
            logger.info(f"Detected NVIDIA CUDA device (GPU count: {torch.cuda.device_count()})")
            return "cuda", -1

        # Check for Intel XPU (Arc GPU)
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            logger.info("Detected Intel XPU device")
            return "xpu", -1
        
        # Fallback to CPU
        else:
            logger.info("No GPU detected, using CPU")
            return "cpu", 0
    except Exception as e:
        logger.warning(f"Error detecting device: {e}, falling back to CPU")
        return "cpu", 0

if __name__ == "__main__":
    device, gpu = get_device()
    print(f"Selected device: {device}, GPU index: {gpu}")