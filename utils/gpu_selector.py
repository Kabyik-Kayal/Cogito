import sys
import os
import platform
import subprocess
import shutil
from utils.logger import get_logger

logger = get_logger(__name__)

def get_device():
    """Auto detect the best available GPU backend for llama-cpp-python
    
    Returns:
        tuple: (backend, n_gpu_layers) where:
            - backend: str - llama-cpp device ('metal', 'cuda', 'sycl', 'vulkan', 'cpu')
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

        # Check for Intel SYCL (oneAPI) - Arc, Data Center, or integrated GPU
        if _has_sycl():
            logger.info("Detected Intel SYCL device")
            return "sycl", -1

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


def _run_command(cmd: list[str], timeout: int = 5) -> tuple[bool, str]:
    """Run a command and return (success, output) - cross-platform compatible"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            # Suppress console window on Windows
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        )
        return result.returncode == 0, result.stdout + result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False, ""


def _has_cuda() -> bool:
    """Check if NVIDIA CUDA is available"""
    # Check for CUDA environment variables first (fast check)
    if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
        return True
    if os.getenv("CUDA_PATH") is not None:
        return True
    
    # Check for nvidia-smi command
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        success, output = _run_command([nvidia_smi])
        if success and "NVIDIA" in output:
            return True
    
    return False


def _has_sycl() -> bool:
    """Check if Intel SYCL (oneAPI) GPU is available"""
    # Check for Intel oneAPI environment variables
    if os.getenv("ONEAPI_ROOT") is not None:
        pass  # Environment suggests oneAPI is installed, continue to verify GPU
    
    # Check for sycl-ls command (lists SYCL devices)
    sycl_ls = shutil.which("sycl-ls")
    if sycl_ls:
        success, output = _run_command([sycl_ls])
        if success:
            # Look for GPU devices in the output (not just CPU)
            output_lower = output.lower()
            if "gpu" in output_lower or "level_zero" in output_lower:
                return True
    
    # Check for Intel GPU via /dev/dri (Linux)
    if platform.system() == "Linux":
        if os.path.exists("/dev/dri/renderD128"):
            # Verify it's Intel by checking for i915 or xe driver
            try:
                with open("/sys/class/drm/renderD128/device/driver/module/drivers", "r") as f:
                    drivers = f.read()
                    if "i915" in drivers or "xe" in drivers:
                        return True
            except (FileNotFoundError, PermissionError):
                # Alternative: check lspci for Intel GPU
                lspci = shutil.which("lspci")
                if lspci:
                    success, output = _run_command([lspci])
                    if success and "Intel" in output and ("VGA" in output or "Display" in output):
                        return True
    
    return False


def _has_vulkan() -> bool:
    """Check if Vulkan is available (fallback for Intel/AMD GPU support)"""
    vulkaninfo = shutil.which("vulkaninfo")
    if vulkaninfo:
        success, output = _run_command([vulkaninfo, "--summary"], timeout=10)
        if success and "GPU" in output:
            return True
    
    # On Windows, check for vulkan-1.dll
    if platform.system() == "Windows":
        system32 = os.path.join(os.environ.get("SYSTEMROOT", "C:\\Windows"), "System32")
        if os.path.exists(os.path.join(system32, "vulkan-1.dll")):
            return True
    
    return False


if __name__ == "__main__":
    backend, n_gpu_layers = get_device()
    print(f"Selected backend: {backend}, GPU layers: {n_gpu_layers}")
    
    # Show detailed detection info
    print("\nDetection details:")
    print(f"  CUDA available: {_has_cuda()}")
    print(f"  SYCL available: {_has_sycl()}")
    print(f"  Vulkan available: {_has_vulkan()}")