import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {getattr(torch.version, 'cuda', 'unknown')}")
print(f"GPU count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Compute capability: {props.major}.{props.minor}")
    print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    x = torch.ones(1000, 1000, device='cuda:0')
    y = x @ x.T
    print(f"CUDA test passed: {y.shape}")
else:
    print("CUDA not available; skipping test.")
