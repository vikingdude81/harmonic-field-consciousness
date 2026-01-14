# Parallel Training Guide - Dual GPU Workflow

**Created:** January 5, 2026
**Hardware:** RTX 5090 (32GB) + RTX A2000 (12GB) + Threadripper 5955WX (16C/32T)
**Purpose:** Run LLM training on GPU 0 while developing sparse eigensolvers on GPU 1

---

## 🎯 Quick Start (After Reboot)

### Terminal 1: RTX 5090 - LLM Training (32B or 350M Model)

```bash
# Navigate to project
cd /mnt/c/Users/akbon/OneDrive/Documents/GitHub/harmonic-field-consciousness/NanoGPT

# Activate environment
source ~/.venv/bin/activate  # Or wherever your venv is

# Start training on GPU 0 (RTX 5090) with 12 CPU threads
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 \
    nohup python train_v5_with_validation.py > training_350M.log 2>&1 &

# Monitor progress
tail -f training_350M.log
```

### Terminal 2: RTX A2000 - Sparse Eigensolver Development

```bash
# Navigate to project
cd /mnt/c/Users/akbon/OneDrive/Documents/GitHub/harmonic-field-consciousness

# Activate environment
source ~/.venv/bin/activate

# Work on eigensolvers using GPU 1 (RTX A2000) with 12 CPU threads
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 \
    python experiments/test_sparse_eigensolvers.py
```

---

## 📋 Pre-Flight Checklist (After Reboot)

- [ ] Boot into Windows 11
- [ ] Start WSL2: Open Ubuntu terminal
- [ ] Verify GPUs visible: `nvidia-smi`
- [ ] Check both GPUs listed:
  - GPU 0: NVIDIA GeForce RTX 5090 (32GB)
  - GPU 1: NVIDIA RTX A2000 (12GB)
- [ ] Navigate to project directory
- [ ] Activate Python virtual environment

---

## 🚀 Detailed Workflow

### Step 1: Verify GPU Setup

```bash
# Check both GPUs are visible
nvidia-smi

# Expected output should show:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 5xx.xx       Driver Version: 5xx.xx       CUDA Version: 12.8   |
# |-------------------------------+----------------------+----------------------+
# |   0  NVIDIA GeForce RTX 5090  |   ...   |   32GB   |
# |   1  NVIDIA RTX A2000         |   ...   |   12GB   |
# +-------------------------------+----------------------+----------------------+
```

### Step 2: Start LLM Training on RTX 5090

```bash
# Open first terminal/tmux pane
cd /mnt/c/Users/akbon/OneDrive/Documents/GitHub/harmonic-field-consciousness/NanoGPT

# Option A: Train 350M model (8-12 hours)
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 \
    nohup python train_350m_stage1.py > training_350M.log 2>&1 &

# Option B: Continue with 32B model stages
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 \
    nohup python train_v5_with_validation.py > training_32B.log 2>&1 &

# Monitor with:
tail -f training_*.log

# Or use watch for live updates:
watch -n 5 'tail -20 training_350M.log'
```

### Step 3: Start Eigensolver Work on RTX A2000

```bash
# Open second terminal/tmux pane
cd /mnt/c/Users/akbon/OneDrive/Documents/GitHub/harmonic-field-consciousness

# Option A: Interactive development (recommended)
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 \
    python experiments/test_sparse_eigensolvers.py

# Option B: Run tests in background
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 \
    nohup python experiments/test_sparse_eigensolvers.py > eigensolver.log 2>&1 &

# Option C: Interactive Python session for development
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 ipython
```

---

## 🎬 Using TMUX for Session Management (Recommended)

```bash
# Install tmux if not already installed
sudo apt install tmux

# Create new tmux session
tmux new -s parallel_work

# Split screen horizontally: Ctrl+b then "
# Now you have two panes

# === Top Pane: LLM Training Monitor ===
cd /mnt/c/Users/akbon/OneDrive/Documents/GitHub/harmonic-field-consciousness/NanoGPT
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 python train_v5_with_validation.py

# Switch to bottom pane: Ctrl+b then ↓

# === Bottom Pane: Eigensolver Development ===
cd /mnt/c/Users/akbon/OneDrive/Documents/GitHub/harmonic-field-consciousness
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 python experiments/test_sparse_eigensolvers.py

# Useful tmux commands:
# - Ctrl+b then ↑/↓: Switch between panes
# - Ctrl+b then d: Detach (keeps running in background)
# - tmux attach -t parallel_work: Re-attach to session
# - Ctrl+b then [: Enter scroll mode (q to exit)
```

---

## 📊 Monitoring Both Tasks

### Check GPU Usage in Real-Time

```bash
# In a third terminal, monitor GPU usage
watch -n 1 nvidia-smi

# Expected output:
# GPU 0 (RTX 5090): ~23GB VRAM, ~95% utilization (LLM training)
# GPU 1 (RTX A2000): ~4-10GB VRAM, ~50-80% utilization (eigensolvers)
```

### Check CPU and RAM Usage

```bash
# Monitor CPU/RAM
htop

# Or use:
top

# Expected:
# CPU: 24-28 out of 32 threads in use (~75%)
# RAM: 60-80GB out of 160GB in use (~50%)
```

### Check Process Status

```bash
# List background processes
ps aux | grep python

# Check if training is running
pgrep -fa train_v5

# Check logs
ls -lh *.log
tail -f training_350M.log
```

---

## ⚙️ Configuration Details

### GPU Pinning

- `CUDA_VISIBLE_DEVICES=0`: Forces process to use GPU 0 (RTX 5090)
- `CUDA_VISIBLE_DEVICES=1`: Forces process to use GPU 1 (RTX A2000)
- Prevents any cross-GPU interference

### CPU Thread Allocation

- `OMP_NUM_THREADS=12`: Allocates 12 CPU threads per process
- 12 + 12 = 24 threads total (out of 32 available)
- Leaves 8 threads for system processes

### Memory Expectations

| Resource | GPU 0 (5090) | GPU 1 (A2000) | Total | Available | Usage % |
|----------|--------------|---------------|-------|-----------|---------|
| GPU VRAM | ~20-25GB | ~4-10GB | ~29GB | 44GB | 66% |
| CPU Threads | 12 | 12 | 24 | 32 | 75% |
| System RAM | ~40-50GB | ~20-30GB | ~70GB | 160GB | 44% |

---

## 🎯 Training Options

### Option 1: 32B Model Training (Original Plan)

```bash
# Terminal 1: RTX 5090
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 \
    python NanoGPT/train_stage3_code.py

# Expected time: ~12 hours
# VRAM: ~23GB
# Quality: GPT-3.5-turbo level
```

### Option 2: 350M Model Training (Recommended)

```bash
# Terminal 1: RTX 5090
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 \
    python NanoGPT/train_350m_stage1.py

# Expected time: ~8-10 hours
# VRAM: ~15-18GB
# Quality: Better than current 113M
```

### Option 3: Dual Model Training

```bash
# Terminal 1: RTX 5090 - 7B model
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 \
    python NanoGPT/train_7b_stage1.py

# Terminal 2: RTX A2000 - 1.3B model
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 \
    python NanoGPT/train_1.3b_stage1.py

# Run two models in parallel!
```

---

## 🛠️ Sparse Eigensolver Development Tasks

### Phase 1: Research and Setup (1-2 hours)

```python
# Test basic sparse eigensolvers on CPU
import numpy as np
from scipy.sparse.linalg import eigsh, eigs
import scipy.sparse as sp

# Small test (1000 nodes)
N = 1000
A = sp.random(N, N, density=0.01, format='csr')
A = (A + A.T) / 2  # Symmetric

# Compute top 10 eigenvalues
vals, vecs = eigsh(A, k=10, which='LM')
print(f"Top eigenvalues: {vals}")
```

### Phase 2: GPU Implementation (2-4 hours)

```python
# Test on RTX A2000 with medium networks
import cupy as cp
import cupyx.scipy.sparse as cpsp
from cupyx.scipy.sparse.linalg import eigsh as cp_eigsh

# Medium test (10,000 nodes)
N = 10000
A = cpsp.random(N, N, density=0.01, format='csr')
A = (A + A.T) / 2

# GPU eigensolver
vals, vecs = cp_eigsh(A, k=10, which='LM')
print(f"Top eigenvalues: {cp.asnumpy(vals)}")
```

### Phase 3: Integration (2-4 hours)

```bash
# Test with actual harmonic bridge code
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 \
    python -c "
from src.neural_mass.harmonic_bridge import HarmonicBridge
import numpy as np

# Create medium network
N = 15000
W = np.random.randn(N, N) * 0.01
W = (W + W.T) / 2

# Test sparse decomposition
bridge = HarmonicBridge(W, use_sparse=True, device='cuda:1')
harmonics = bridge.get_harmonics(k=50)
print(f'Computed {len(harmonics)} harmonics')
"
```

---

## 🔍 Troubleshooting

### Issue: GPU not detected

```bash
# Check driver
nvidia-smi

# Restart WSL if needed
wsl --shutdown
# Then reopen WSL terminal
```

### Issue: Out of memory on A2000

```bash
# Reduce network size for eigensolvers
# Or use CPU instead:
python experiments/test_sparse_eigensolvers.py --device cpu
```

### Issue: Training process killed

```bash
# Check logs
tail -100 training_*.log

# Check system memory
free -h

# May need to reduce batch size or use gradient checkpointing
```

### Issue: Can't see both GPUs

```bash
# Verify in Windows Device Manager first
# Then check WSL can see them:
nvidia-smi

# If only one GPU visible, may need to update WSL/drivers
```

---

## 📈 Expected Timeline

| Time | GPU 0 (RTX 5090) | GPU 1 (RTX A2000) |
|------|------------------|-------------------|
| **Hour 0-1** | Training startup, warmup | Research sparse methods |
| **Hour 1-3** | Training in progress | Implement sparse solver |
| **Hour 3-6** | Training in progress | Test on small networks (CPU) |
| **Hour 6-9** | Training in progress | Test on medium networks (GPU) |
| **Hour 9-12** | Training completing | Integration with harmonic bridge |
| **Hour 12+** | Validation & testing | Large-scale testing on RTX 5090 |

---

## ✅ Success Criteria

### LLM Training Success:
- [ ] Training completes without OOM errors
- [ ] Validation loss decreases steadily
- [ ] Final model checkpoint saved
- [ ] Model quality assessed (perplexity < 10)

### Eigensolver Development Success:
- [ ] Sparse solver works on 5K-25K node networks
- [ ] Results match dense solver (for small test cases)
- [ ] GPU acceleration functional
- [ ] Integration with HarmonicBridge complete
- [ ] Unit tests pass

---

## 📝 Notes

- **Total expected time:** 8-12 hours for parallel workflow
- **System load:** ~70% GPU, ~75% CPU, ~50% RAM (comfortable)
- **Calendar time savings:** ~8-10 hours vs sequential approach
- **Productivity:** Both major tasks advance simultaneously

---

## 🎓 Commands Reference

```bash
# Quick reference card

# === GPU Management ===
nvidia-smi                          # Check GPU status
watch -n 1 nvidia-smi              # Live GPU monitoring
CUDA_VISIBLE_DEVICES=0 <cmd>       # Use GPU 0 only
CUDA_VISIBLE_DEVICES=1 <cmd>       # Use GPU 1 only

# === Process Management ===
nohup <cmd> > log.txt 2>&1 &       # Run in background
pgrep -fa python                    # List Python processes
pkill -f train_                     # Kill training processes
tail -f log.txt                     # Monitor log file

# === TMUX Management ===
tmux new -s name                    # New session
tmux ls                             # List sessions
tmux attach -t name                 # Attach to session
Ctrl+b then d                       # Detach session
Ctrl+b then "                       # Split horizontal
Ctrl+b then ↑/↓                     # Switch panes

# === System Monitoring ===
htop                                # CPU/RAM monitor
free -h                             # Memory usage
df -h                               # Disk usage
```

---

## 🚦 When You Return from Reboot

1. Open this file: `PARALLEL_TRAINING_GUIDE.md`
2. Follow "Quick Start" section at the top
3. Use TMUX workflow for easy monitoring
4. Check "Monitoring Both Tasks" section periodically
5. Refer to "Troubleshooting" if issues arise

**Good luck with your parallel training!** 🚀

---

**Last Updated:** January 5, 2026
**Status:** Ready to use after reboot
**Estimated Total Time:** 8-12 hours (parallel execution)
