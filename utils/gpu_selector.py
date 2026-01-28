import sys
import torch

def get_device():
    """Auto detect the best available GPU backend"""

    try:
        # Check for Apple Silicon
        if torch.mps.is_available():
            return "Metal (Apple Silicon)", -1
        
        # Check for Nvidia GPU
        elif torch.cuda.is_available():
            return "Nvidia CUDA", -1

        # Check for Intel Arc
        elif torch.xpu.is_available():
            return "Intel XPU", -1
        
        # Fallback to CPU
        else:
            return "CPU", 0
    except Exception as e:
        logger.info(f"Error : {e}",sys)
        return "CPU",0

if __name__ == "__main__":
    device, gpu = get_device()
    print(device,gpu)