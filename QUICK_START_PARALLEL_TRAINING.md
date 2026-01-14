# Quick Start: Parallel Training

**Goal:** Run LLM training on GPU 0 while working on eigensolvers on GPU 1

---

## 🚀 Step 1: Start LLM Training (WSL2)

```bash
# Open WSL2 Ubuntu terminal
wsl

# Navigate to project
cd /mnt/c/Users/akbon/OneDrive/Documents/GitHub/harmonic-field-consciousness

# Run startup script
./start_parallel_training.sh
```

**Expected Output:**
```
Training started with PID: 12345
✓ Training process is running!
Monitor with: tail -f .../NanoGPT/training_100k.log
```

**Duration:** ~15 hours (hands-off)

---

## 🔧 Step 2: Work on Eigensolvers (Windows or WSL2)

While training runs, work on eigensolvers:

```bash
# Test sparse eigensolvers
python experiments/test_sparse_eigensolvers.py --sizes 25000 30000 50000

# Run demonstrations
python experiments/demo_sparse_harmonics.py --all --nodes 30000

# Integrate with existing experiments
python experiments/category2_dynamics/test_with_sparse_solver.py
```

---

## 📊 Monitor Both Tasks

### Check GPU Usage:
```bash
# WSL2 or Windows
nvidia-smi

# Continuous monitoring
watch -n 5 nvidia-smi
```

**Expected:**
- GPU 0: ~23GB VRAM, ~95% utilization (training)
- GPU 1: Variable usage (your development work)

### Check Training Progress:
```bash
# WSL2
tail -f /mnt/c/Users/akbon/OneDrive/Documents/GitHub/harmonic-field-consciousness/NanoGPT/training_100k.log

# Look for:
# Step 100: loss 0.xxx
# Step 500: checkpoint saved
```

---

## ⏸️ Stop/Resume Training

### Stop Training:
```bash
# Find process ID
pgrep -fa train_100k_production

# Kill process
pkill -f train_100k_production
```

### Resume Training:
Training auto-saves checkpoints every 500 steps. To resume:
```bash
# Edit train_100k_production.py to load from latest checkpoint
# Or use checkpoint path in model loading
```

---

## ✅ When Training Completes

After ~15 hours:

```bash
# Check final model
ls -lh NanoGPT/outputs_100k/final/

# Validate model quality
python NanoGPT/validate_model.py

# Test generation
python NanoGPT/generate_samples.py
```

---

## 🎯 Next Session Priorities

1. ✅ Training complete → Validate model quality
2. ⏳ Re-run GPU experiments with fixed randomization
3. ⏳ Add repetition penalty to generation
4. ⏳ Scale to 350M parameter model

---

## 📚 Documentation

- **Full Guide:** [PARALLEL_TRAINING_GUIDE.md](PARALLEL_TRAINING_GUIDE.md)
- **Session Summary:** [PARALLEL_TRAINING_SESSION_SUMMARY.md](PARALLEL_TRAINING_SESSION_SUMMARY.md)
- **Eigensolver Docs:** `src/neural_mass/sparse_harmonic_bridge.py` (docstrings)

---

## 🆘 Troubleshooting

**Training not starting?**
- Check: `python3 -c "import unsloth; print('OK')"`
- If error: Activate correct conda/venv environment

**GPU not visible?**
- Check: `nvidia-smi` shows both GPUs
- If not: Restart WSL with `wsl --shutdown`

**Out of memory?**
- Training uses ~23GB (RTX 5090 has 32GB - should be fine)
- If OOM: Reduce batch size in script

---

**Ready to start? Run:** `./start_parallel_training.sh` **in WSL2!**
