"""
Test HarmonicGPT V6 Improvements
================================

Validate that V6 improvements are working correctly:
1. Consciousness metrics computation
2. Rotation angle monitoring
3. Wave pattern detection
4. Stochastic depth
5. Activation noise
6. Consciousness regularization loss

This script tests the model WITHOUT full training - just forward passes.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add NanoGPT to path
sys.path.insert(0, 'NanoGPT')

from harmonic_model_v6 import HarmonicGPTV6

print("=" * 80)
print("TESTING HARMONICGPT V6 IMPROVEMENTS")
print("=" * 80)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")

# Initialize model with V6 features
print("\n[1/7] Initializing HarmonicGPT V6...")
config = HarmonicGPTV6.get_default_config()
config.update({
    'n_layer': 4,
    'n_head': 4,
    'n_embd': 128,
    'block_size': 64,
    'vocab_size': 50304,
    'dropout': 0.1,
    'stochastic_depth_rate': 0.1,
    'activation_noise_std': 0.01,
    'consciousness_loss_weight': 0.01,
    'target_consciousness': 0.25,
})

model = HarmonicGPTV6(config)
model = model.to(device)
model.eval()

print("[OK] Model initialized")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Stochastic depth rate: {model.stochastic_depth_rate}")
print(f"  Activation noise std: {model.activation_noise_std}")
print(f"  Consciousness loss weight: {model.consciousness_loss_weight}")

# Test 1: Basic forward pass
print("\n[2/7] Testing basic forward pass...")
batch_size = 2
seq_len = 32
x = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
y = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)

with torch.no_grad():
    logits, loss, c_loss = model(x, y)

print("[OK] Forward pass successful")
print(f"  Logits shape: {logits.shape}")
print(f"  CE Loss: {loss.item():.4f}")
print(f"  Consciousness Loss: {c_loss:.4f}")

# Test 2: Consciousness metrics computation
print("\n[3/7] Testing consciousness metrics computation...")
with torch.no_grad():
    logits, loss, metrics = model(x, y, output_consciousness_metrics=True)

print("[OK] Metrics computed")
print(f"  Rotation angle: {metrics['rotation']:.1f}° (target: 1500-4500°)")
print(f"  Has wave pattern: {metrics['has_wave']}")
print(f"  Diversity (CV): {metrics['diversity']:.3f} (target: >0.5)")
print(f"  Estimated consciousness: {metrics['estimated_consciousness']:.3f} (target: 0.25)")

# Validate ranges
checks = []
checks.append(("Rotation in range", target_rotation_min <= metrics['rotation'] <= target_rotation_max))
checks.append(("Diversity acceptable", metrics['diversity'] > 0.1))
checks.append(("Consciousness in range", 0.2 <= metrics['estimated_consciousness'] <= 0.4))

for check_name, passed in checks:
    status = "[OK]" if passed else "[WARN]"
    print(f"  {status} {check_name}")

# Test 3: Rotation angle diversity across batches
print("\n[4/7] Testing rotation angle diversity...")
rotations = []
for i in range(10):
    x_test = torch.randint(0, config['vocab_size'], (1, seq_len), device=device)
    with torch.no_grad():
        _, _, metrics = model(x_test, output_consciousness_metrics=True)
    rotations.append(metrics['rotation'])

rotation_mean = np.mean(rotations)
rotation_std = np.std(rotations)
rotation_cv = rotation_std / rotation_mean if rotation_mean > 0 else 0

print(f"[OK] Rotation statistics over 10 batches:")
print(f"  Mean: {rotation_mean:.1f}°")
print(f"  Std:  {rotation_std:.1f}°")
print(f"  CV:   {rotation_cv:.3f}")

if rotation_cv > 0.3:
    print(f"  [OK] High diversity (CV > 0.3)")
elif rotation_cv > 0.1:
    print(f"  [WARN] Moderate diversity (0.1 < CV < 0.3)")
else:
    print(f"  [FAIL] Low diversity (CV < 0.1)")

# Test 4: Stochastic depth (training mode)
print("\n[5/7] Testing stochastic depth in training mode...")
model.train()

outputs_train = []
for i in range(5):
    with torch.no_grad():
        logits, _, _ = model(x)
    outputs_train.append(logits[0, 0, :10].cpu().numpy())

# Check if outputs vary (stochastic depth is working)
outputs_train = np.array(outputs_train)
variance_across_runs = np.var(outputs_train, axis=0).mean()

print(f"[OK] Stochastic depth tested")
print(f"  Variance across 5 runs: {variance_across_runs:.6f}")

if variance_across_runs > 1e-6:
    print(f"  [OK] Outputs vary (stochastic depth active)")
else:
    print(f"  [WARN] Outputs identical (may need more layers or higher rate)")

model.eval()

# Test 5: Wave pattern detection
print("\n[6/7] Testing wave pattern detection...")
wave_detected_count = 0
total_tests = 20

for i in range(total_tests):
    x_test = torch.randint(0, config['vocab_size'], (1, seq_len), device=device)
    with torch.no_grad():
        _, _, metrics = model(x_test, output_consciousness_metrics=True)
    if metrics['has_wave']:
        wave_detected_count += 1

wave_pct = (wave_detected_count / total_tests) * 100

print(f"[OK] Wave detection statistics over {total_tests} batches:")
print(f"  Waves detected: {wave_detected_count}/{total_tests} ({wave_pct:.1f}%)")
print(f"  Target: ~50% (balanced)")

if 30 <= wave_pct <= 70:
    print(f"  [OK] Balanced wave detection")
elif 10 <= wave_pct <= 90:
    print(f"  [WARN] Moderate imbalance")
else:
    print(f"  [FAIL] Severe imbalance")

# Test 6: Consciousness regularization loss
print("\n[7/7] Testing consciousness regularization loss...")
model.train()

# Create batch with target labels
x_train = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
y_train = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)

with torch.no_grad():
    logits, ce_loss, c_loss = model(x_train, y_train)

print(f"[OK] Loss components:")
print(f"  CE Loss: {ce_loss.item():.4f}")
print(f"  Consciousness Loss: {c_loss:.4f}")
print(f"  Total Loss (with λ={model.consciousness_loss_weight}): {ce_loss.item() + model.consciousness_loss_weight * c_loss:.4f}")

if c_loss > 0:
    print(f"  [OK] Consciousness loss is non-zero")
else:
    print(f"  [WARN] Consciousness loss is zero (may need adjustment)")

# Test 7: Generation with consciousness tracking
print("\n" + "=" * 80)
print("BONUS: Testing generation with consciousness tracking")
print("=" * 80)

model.eval()
prompt = torch.tensor([[1, 2, 3]], device=device)  # Dummy prompt

print("\nGenerating 20 tokens with consciousness tracking...")
with torch.no_grad():
    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_k=40,
        repetition_penalty=1.0  # V6 should inherit this from V5
    )

print(f"[OK] Generated {generated.shape[1]} tokens")
print(f"  Shape: {generated.shape}")

# Measure consciousness on generated sequence
with torch.no_grad():
    _, _, metrics = model(generated, output_consciousness_metrics=True)

print(f"\nConsciousness metrics on generated sequence:")
print(f"  Rotation: {metrics['rotation']:.1f}°")
print(f"  Wave pattern: {metrics['has_wave']}")
print(f"  Diversity: {metrics['diversity']:.3f}")
print(f"  Estimated consciousness: {metrics['estimated_consciousness']:.3f}")

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

all_tests = [
    ("Basic forward pass", True),
    ("Consciousness metrics", True),
    ("Rotation diversity", rotation_cv > 0.1),
    ("Stochastic depth", variance_across_runs > 1e-6),
    ("Wave detection", 10 <= wave_pct <= 90),
    ("Consciousness loss", c_loss > 0),
    ("Generation", generated.shape[1] > prompt.shape[1]),
]

passed_tests = sum(1 for _, passed in all_tests if passed)
total_tests = len(all_tests)

print(f"\nTests passed: {passed_tests}/{total_tests}")
for test_name, passed in all_tests:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} {test_name}")

if passed_tests == total_tests:
    print("\n[SUCCESS] All tests passed! V6 improvements are working correctly.")
elif passed_tests >= total_tests * 0.7:
    print("\n[PARTIAL] Most tests passed. Some improvements may need tuning.")
else:
    print("\n[FAIL] Several tests failed. V6 improvements need debugging.")

print("\n" + "=" * 80)
print("Next steps:")
print("  1. Train V6 model: cd NanoGPT && python train_v6_consciousness_aware.py")
print("  2. Compare with baseline: python validate_nanogpt_consciousness.py")
print("  3. Measure improvements: see NANOGPT_IMPROVEMENTS.md")
print("=" * 80)

target_rotation_min = 1500
target_rotation_max = 4500
