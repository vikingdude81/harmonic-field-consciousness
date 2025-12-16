"""
GPU/CUDA Utilities

Provides GPU-accelerated computation when available.
Falls back to NumPy/CPU if CUDA is not available.

Supports:
- CuPy for general GPU array operations
- PyTorch for neural network experiments
- Automatic device detection and fallback
"""

import numpy as np
from typing import Optional, Tuple, Union, Any
import warnings

# Try to import GPU libraries
CUPY_AVAILABLE = False
TORCH_AVAILABLE = False
TORCH_CUDA_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None

try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None


def get_array_module(use_gpu: bool = True):
    """
    Get the appropriate array module (CuPy or NumPy).
    
    Args:
        use_gpu: Whether to use GPU if available
        
    Returns:
        Array module (cupy or numpy)
    """
    if use_gpu and CUPY_AVAILABLE:
        return cp
    return np


def to_gpu(arr: np.ndarray) -> Any:
    """
    Move array to GPU if available.
    
    Args:
        arr: NumPy array
        
    Returns:
        CuPy array if GPU available, else original array
    """
    if CUPY_AVAILABLE:
        return cp.asarray(arr)
    return arr


def to_cpu(arr: Any) -> np.ndarray:
    """
    Move array to CPU.
    
    Args:
        arr: Array (NumPy or CuPy)
        
    Returns:
        NumPy array
    """
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def get_device_info() -> dict:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cupy_available': CUPY_AVAILABLE,
        'torch_available': TORCH_AVAILABLE,
        'torch_cuda_available': TORCH_CUDA_AVAILABLE,
        'gpu_devices': [],
        'recommended_backend': 'cpu'
    }
    
    if CUPY_AVAILABLE:
        try:
            device = cp.cuda.Device()
            info['gpu_devices'].append({
                'id': device.id,
                'name': cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode(),
                'memory_total': cp.cuda.runtime.getDeviceProperties(device.id)['totalGlobalMem'],
            })
            info['recommended_backend'] = 'cupy'
        except Exception as e:
            warnings.warn(f"CuPy available but GPU error: {e}")
    
    if TORCH_CUDA_AVAILABLE:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info['gpu_devices'].append({
                'id': i,
                'name': props.name,
                'memory_total': props.total_memory,
                'compute_capability': (props.major, props.minor)
            })
        if not CUPY_AVAILABLE:
            info['recommended_backend'] = 'torch'
    
    return info


class GPUAccelerator:
    """
    Context manager for GPU-accelerated computations.
    
    Usage:
        with GPUAccelerator() as xp:
            # xp is cupy if GPU available, else numpy
            result = xp.sum(xp.array([1, 2, 3]))
    """
    
    def __init__(self, use_gpu: bool = True, device_id: int = 0):
        """
        Initialize GPU accelerator.
        
        Args:
            use_gpu: Whether to use GPU if available
            device_id: GPU device ID to use
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.device_id = device_id
        self.xp = cp if self.use_gpu else np
        
    def __enter__(self):
        if self.use_gpu:
            cp.cuda.Device(self.device_id).use()
        return self.xp
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()
        return False


def gpu_eigendecomposition(matrix: np.ndarray, use_gpu: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated eigendecomposition.
    
    Args:
        matrix: Square matrix for eigendecomposition
        use_gpu: Whether to use GPU
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    if use_gpu and CUPY_AVAILABLE:
        matrix_gpu = cp.asarray(matrix)
        eigenvalues, eigenvectors = cp.linalg.eigh(matrix_gpu)
        return cp.asnumpy(eigenvalues), cp.asnumpy(eigenvectors)
    else:
        return np.linalg.eigh(matrix)


def gpu_matrix_multiply(a: np.ndarray, b: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    """
    GPU-accelerated matrix multiplication.
    
    Args:
        a: First matrix
        b: Second matrix
        use_gpu: Whether to use GPU
        
    Returns:
        Result matrix
    """
    if use_gpu and CUPY_AVAILABLE:
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        result = cp.matmul(a_gpu, b_gpu)
        return cp.asnumpy(result)
    else:
        return np.matmul(a, b)


def gpu_fft(arr: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    """
    GPU-accelerated FFT.
    
    Args:
        arr: Input array
        use_gpu: Whether to use GPU
        
    Returns:
        FFT result
    """
    if use_gpu and CUPY_AVAILABLE:
        arr_gpu = cp.asarray(arr)
        result = cp.fft.fft(arr_gpu)
        return cp.asnumpy(result)
    else:
        return np.fft.fft(arr)


def batch_compute_metrics_gpu(
    powers: np.ndarray,
    eigenvalues: np.ndarray,
    use_gpu: bool = True
) -> dict:
    """
    Batch compute consciousness metrics on GPU.
    
    Args:
        powers: Array of power distributions (n_samples, n_modes)
        eigenvalues: Eigenvalues array
        use_gpu: Whether to use GPU
        
    Returns:
        Dictionary of metric arrays
    """
    xp = get_array_module(use_gpu)
    
    if use_gpu and CUPY_AVAILABLE:
        powers = xp.asarray(powers)
        eigenvalues = xp.asarray(eigenvalues)
    
    n_samples, n_modes = powers.shape
    
    # Normalize powers
    powers = powers / (powers.sum(axis=1, keepdims=True) + 1e-12)
    
    # Mode entropy (vectorized)
    log_powers = xp.log(powers + 1e-12)
    H_mode = -xp.sum(powers * log_powers, axis=1)
    H_max = xp.log(n_modes)
    H_mode = H_mode / (H_max + 1e-12)
    
    # Participation ratio (vectorized)
    PR = 1.0 / (xp.sum(powers ** 2, axis=1) + 1e-12)
    PR = PR / n_modes
    
    # Criticality index (vectorized) - simplified version
    mean_power = powers.mean(axis=1, keepdims=True)
    variance = xp.mean((powers - mean_power) ** 2, axis=1)
    kappa = variance / (mean_power.squeeze() ** 2 + 1e-12)
    kappa = xp.tanh(kappa)  # Normalize to [0, 1]
    
    # Convert back to numpy if needed
    if use_gpu and CUPY_AVAILABLE:
        H_mode = cp.asnumpy(H_mode)
        PR = cp.asnumpy(PR)
        kappa = cp.asnumpy(kappa)
    
    return {
        'H_mode': H_mode,
        'PR': PR,
        'kappa': kappa,
    }


def print_gpu_status():
    """Print GPU availability status."""
    info = get_device_info()
    
    print("=" * 60)
    print("GPU Status")
    print("=" * 60)
    print(f"CuPy available:      {info['cupy_available']}")
    print(f"PyTorch available:   {info['torch_available']}")
    print(f"PyTorch CUDA:        {info['torch_cuda_available']}")
    print(f"Recommended backend: {info['recommended_backend']}")
    
    if info['gpu_devices']:
        print("\nGPU Devices:")
        for dev in info['gpu_devices']:
            mem_gb = dev.get('memory_total', 0) / (1024**3)
            print(f"  [{dev['id']}] {dev['name']} ({mem_gb:.1f} GB)")
    else:
        print("\nNo GPU devices detected. Using CPU.")
    print("=" * 60)


if __name__ == "__main__":
    print_gpu_status()
