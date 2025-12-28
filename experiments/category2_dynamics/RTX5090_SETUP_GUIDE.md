# RTX 5090 (sm_120) GPU Setup Guide for Harmonic Field Experiments

## Overview

The NVIDIA RTX 5090 uses the Blackwell architecture with CUDA compute capability **sm_120** (CUDA 12.0). This requires PyTorch nightly builds with CUDA 12.8+ support, as stable PyTorch releases (as of Dec 2025) only support up to sm_90.

## Hardware Specifications

| Spec | RTX 5090 |
|------|----------|
| Architecture | Blackwell (sm_120) |
| VRAM | 32 GB GDDR7 |
| CUDA Cores | 21,760 |
| Tensor Cores | 680 |
| Memory Bandwidth | 1.79 TB/s |

## Prerequisites

- Windows 10/11 or Linux with latest NVIDIA drivers (560+)
- Python 3.11 (recommended) or 3.10
- CUDA Toolkit 12.8+ (driver includes runtime)

## Installation Steps

### Step 1: Install Latest NVIDIA Drivers

Download from: https://www.nvidia.com/download/index.aspx

Verify installation:
```powershell
nvidia-smi
```

Expected output should show:
- Driver Version: 560.xx or higher
- CUDA Version: 12.8 or higher

### Step 2: Install Python 3.11

Download from: https://www.python.org/downloads/

Install to: `C:\Users\<username>\AppData\Local\Programs\Python\Python311\`

Verify:
```powershell
& "C:\Users\<username>\AppData\Local\Programs\Python\Python311\python.exe" --version
```

### Step 3: Install PyTorch Nightly with CUDA 12.8

**CRITICAL**: Stable PyTorch does NOT support sm_120. You MUST use nightly builds.

```powershell
# Uninstall existing PyTorch
& "C:\Users\<username>\AppData\Local\Programs\Python\Python311\python.exe" -m pip uninstall torch torchvision torchaudio -y

# Install PyTorch nightly with CUDA 12.8
& "C:\Users\<username>\AppData\Local\Programs\Python\Python311\python.exe" -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Step 4: Verify Installation

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    props = torch.cuda.get_device_properties(i)
    print(f"  Compute capability: {props.major}.{props.minor}")
    print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")

# Test CUDA operation
x = torch.ones(1000, 1000, device='cuda:0')
y = x @ x.T
print(f"\nCUDA test passed: {y.shape}")
```

Expected output:
```
PyTorch version: 2.11.0.dev20251226+cu128
CUDA available: True
CUDA version: 12.8
GPU count: 2
GPU 0: NVIDIA GeForce RTX 5090
  Compute capability: 12.0
  Total memory: 34.19 GB
GPU 1: NVIDIA RTX A2000 12GB
  Compute capability: 8.6
  Total memory: 12.88 GB

CUDA test passed: torch.Size([1000, 1000])
```

### Step 5: Install Additional Dependencies

```powershell
& "C:\Users\<username>\AppData\Local\Programs\Python\Python311\python.exe" -m pip install numpy scipy pandas tqdm matplotlib
```

## GPU Selection in Code

When you have multiple GPUs, always explicitly select GPU 0 (RTX 5090):

```python
import torch

# GPU Selection - Prefer GPU 0 (RTX 5090) for heavy CUDA work
if torch.cuda.is_available():
    print("Available GPUs:")
    for gpu_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(gpu_id)
        print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)} ({props.total_memory / 1e9:.2f} GB)")
    
    # Explicitly use GPU 0 (RTX 5090) for heavy computation
    preferred_gpu = 0
    device = torch.device(f'cuda:{preferred_gpu}')
    print(f"\nUsing: GPU {preferred_gpu} - {torch.cuda.get_device_name(preferred_gpu)}")
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")
```

## Eigendecomposition Limits

### cuSOLVER Dense Eigendecomposition

The `torch.linalg.eigh()` function uses cuSOLVER for GPU-accelerated eigendecomposition. There are practical limits:

| Matrix Size | Nodes | Status | Eigendecomp Time | Notes |
|-------------|-------|--------|------------------|-------|
| 10,000 × 10,000 | 10,000 | ✅ Works | ~2s | Fast |
| 24,964 × 24,964 | 24,964 | ✅ Works | ~13.6s | Safe maximum |
| 25,921 × 25,921 | 25,921 | ✅ Works | ~14.9s | **ABSOLUTE MAXIMUM** |
| 26,896 × 26,896 | 26,896 | ❌ Fails | N/A | cuSOLVER error |
| 29,929 × 29,929 | 29,929 | ❌ Fails | N/A | cuSOLVER error |

### Error When Exceeding Limit

```
torch._C._LinAlgError: cusolver error: CUSOLVER_STATUS_INVALID_VALUE, 
when calling `cusolverDnXsyevd_bufferSize(...)`. 
This error may appear if the input matrix contains NaN.
```

**Note**: The error message about NaN is misleading. The actual issue is exceeding cuSOLVER's matrix size limit for the syevd algorithm.

### Workarounds for Larger Networks

For networks > 26,000 nodes:

1. **Iterative Methods (Lanczos/ARPACK)**:
   ```python
   from scipy.sparse.linalg import eigsh
   # Only compute first k eigenvectors
   eigenvalues, eigenvectors = eigsh(laplacian_sparse, k=2000, which='SM')
   ```

2. **Block Decomposition**: Partition the network and compute local eigenmodes

3. **CPU Fallback**: Use NumPy/SciPy on CPU (slower but no size limit)

4. **Randomized SVD**: For approximate eigenmodes

## Experiment Configurations

### Verified Working Configurations on RTX 5090

| Config | Nodes | Modes | Timesteps | Trials | Eigendecomp | Trial Time | Total |
|--------|-------|-------|-----------|--------|-------------|------------|-------|
| small | 961 | 100 | 1,000 | 100 | 0.1s | 0.07s | 7s |
| medium | 2,401 | 300 | 2,000 | 500 | 0.3s | 0.02s | 10s |
| large | 4,900 | 800 | 5,000 | 200 | 0.8s | 0.1s | 20s |
| xlarge | 10,000 | 1,500 | 10,000 | 100 | 2s | 0.4s | 40s |
| mega | 24,964 | 2,000 | 10,000 | 50 | 13.6s | 1.1s | 55s |
| giga | 24,964 | 2,000 | 15,000 | 50 | 13.6s | 1.2s | 60s |
| ultra | 25,921 | 2,200 | 15,000 | 40 | 14.9s | 1.25s | 50s |
| max | 25,921 | 2,500 | 20,000 | 100 | 14.9s | ~1.5s | ~150s |

### Memory Usage Estimates

| Config | Laplacian | Eigenvectors | Trajectory | Total Peak |
|--------|-----------|--------------|------------|------------|
| small | 7.4 MB | 0.4 MB | 0.4 MB | ~50 MB |
| medium | 46 MB | 2.9 MB | 2.3 MB | ~200 MB |
| large | 192 MB | 16 MB | 12 MB | ~500 MB |
| xlarge | 800 MB | 60 MB | 48 MB | ~2 GB |
| mega | 5.0 GB | 200 MB | 240 MB | ~8 GB |
| ultra | 5.4 GB | 228 MB | 264 MB | ~9 GB |

RTX 5090's 34 GB VRAM provides ample headroom for all configurations.

## Troubleshooting

### Issue: "no kernel image is available for execution on the device"

**Cause**: PyTorch stable release doesn't support sm_120

**Solution**: Install PyTorch nightly with CUDA 12.8:
```powershell
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Issue: "CUSOLVER_STATUS_INVALID_VALUE"

**Cause**: Matrix too large for cuSOLVER eigendecomposition

**Solution**: Reduce network size to ≤25,921 nodes or use iterative methods

### Issue: Wrong GPU selected

**Cause**: PyTorch defaulting to first "compatible" GPU

**Solution**: Explicitly set `device = torch.device('cuda:0')` for RTX 5090

### Issue: Out of Memory

**Cause**: Batch size too large for available VRAM

**Solution**: Reduce batch_size in config (try 2 or 1)

## Running Experiments

```powershell
# Navigate to experiments directory
cd "c:\Users\akbon\OneDrive\Documents\GitHub\harmonic-field-consciousness\experiments\category2_dynamics"

# Run with system Python 3.11 (has PyTorch nightly)
& "C:\Users\akbon\AppData\Local\Programs\Python\Python311\python.exe" exp_gpu_massive_batched.py <config>

# Available configs: small, medium, large, xlarge, mega, giga, ultra, max
```

## Performance Benchmarks

### RTX 5090 vs Previous Generation

| Metric | RTX 4090 | RTX 5090 | Improvement |
|--------|----------|----------|-------------|
| Max eigendecomp nodes | ~20,000 | ~26,000 | +30% |
| Eigendecomp 25K | ~20s | ~14s | 1.4x faster |
| Trajectory sim/s | ~40K | ~60K | 1.5x faster |
| VRAM | 24 GB | 32 GB | +33% |

### Throughput by Configuration

| Config | Trials/sec | Modes×Timesteps/sec |
|--------|------------|---------------------|
| small | 14.3 | 1.43M |
| medium | 50.0 | 30.0M |
| large | 10.0 | 40.0M |
| xlarge | 2.5 | 37.5M |
| mega | 0.9 | 18.0M |
| ultra | 0.8 | 26.4M |

## Scientific Implications

### Scale Advantages

At 25,000+ nodes, we can simulate:

1. **Brain-Scale Networks**: 25K nodes approaches the scale of macroscopic brain parcellations (e.g., Schaefer atlas with 1000 parcels, replicated across 25 subjects)

2. **High-Resolution Spatial Dynamics**: 161×161 lattice provides ~0.6% spatial resolution, suitable for:
   - Fine-grained wave propagation analysis
   - Multi-scale integration/differentiation studies
   - Critical dynamics near phase transitions

3. **Long Trajectories**: 15,000-20,000 timesteps enable:
   - Slow dynamics characterization (10x longer than typical fMRI TRs)
   - State transition detection over extended periods
   - Attractor landscape exploration

4. **Statistical Power**: 50-100 trials per configuration provides:
   - Robust mean/variance estimates
   - Rare event detection (wave formation ~25% of trials)
   - Condition comparison with adequate effect sizes

### Recommended Experimental Protocol

For publication-quality results:

1. **Baseline**: Run `mega` config (50 trials, 25K nodes)
2. **High-Resolution**: Run `ultra` config (40 trials, 26K nodes)
3. **Extended Dynamics**: Run `max` config (100 trials, 20K timesteps)
4. **Scaling Analysis**: Run all configs and compare metrics vs. network size

---

*Last updated: December 27, 2025*
*Tested on: NVIDIA GeForce RTX 5090 (34.19 GB), PyTorch 2.11.0.dev20251226+cu128*
