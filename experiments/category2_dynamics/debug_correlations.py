"""
Debug: What do the correlations actually look like?
"""
import torch
import numpy as np

# Simulate different activity patterns
torch.manual_seed(42)
timesteps = 100
n_nodes = 100

# Type 0: Gaussian blob that decays
gaussian = torch.randn(n_nodes) * torch.linspace(1.0, 0.1, timesteps).unsqueeze(1)
gaussian = gaussian.T  # (timesteps, n_nodes)

# Type 3: Random noise
random_noise = torch.randn(timesteps, n_nodes)

print("="*60)
print("TESTING CORRELATION PATTERNS")
print("="*60)

for name, activity in [("Gaussian (decaying)", gaussian), ("Random Noise", random_noise)]:
    print(f"\n{name}:")
    print(f"  Shape: {activity.shape}")
    
    correlations = []
    for lag in range(1, 20):
        act_early = activity[:-lag].flatten()
        act_late = activity[lag:].flatten()
        corr = torch.corrcoef(torch.stack([act_early, act_late]))[0, 1]
        if not torch.isnan(corr):
            correlations.append(corr.item())
    
    mean_early = sum(correlations[:5]) / 5
    mean_late = sum(correlations[-5:]) / 5
    
    print(f"  Early correlations (lag 1-5): {correlations[:5]}")
    print(f"  Mean early: {mean_early:.3f}")
    print(f"  Late correlations (lag 15-19): {correlations[-5:]}")
    print(f"  Mean late: {mean_late:.3f}")
    print(f"  Has wave? {mean_early > 0.3 and mean_early > mean_late}")

print("\n" + "="*60)
print("INSIGHT:")
print("="*60)
print("""
Random noise at t and t+1 are INDEPENDENT:
  - corr(noise[t], noise[t+1]) ≈ 0
  
But we're correlating FLATTENED arrays:
  - act_early = activity[:-lag].flatten()  # (timesteps-lag)*nodes
  - act_late = activity[lag:].flatten()
  
This correlates:
  - activity[0,:]  with activity[lag,:]
  - activity[1,:]  with activity[lag+1,:]
  - activity[2,:]  with activity[lag+2,:]
  ...
  
For random noise, these ARE independent → correlation ≈ 0 ✓
For structured patterns, spatial structure may persist → correlation > 0

Let me test with SPATIAL positions...
""")
