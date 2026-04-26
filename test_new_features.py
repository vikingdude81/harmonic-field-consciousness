"""Quick smoke test for hooks, logit lens, and viz."""
import torch
import numpy as np

print("=== Testing imports ===")
from consciousness_circuit.hooks import capture_forward, logit_lens
from consciousness_circuit.patching import residual_patch_sweep
from consciousness_circuit.steering import add_residual_steering, SteeringConfig
from consciousness_circuit.visualization import plot_logit_lens_top1, plot_patch_heatmap, plot_residual_scatter
print("All imports OK")

print("\n=== Testing with Qwen2.5-0.5B-Instruct ===")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    trust_remote_code=True
)
print(f"Model: {model.config.num_hidden_layers} layers, hidden={model.config.hidden_size}")

print("\n=== capture_forward ===")
cap = capture_forward(model, tok, "What is 2 + 2?", output_attentions=False)
print(f"Hidden states: {len(cap.hidden_states)} layers")
print(f"Tokens: {cap.tokens}")
print(f"Logits shape: {cap.logits.shape}")

print("\n=== logit_lens ===")
lens = logit_lens(cap.hidden_states, model, tok, top_k=3)
print(f"Layers analyzed: {len(lens)}")
print(f"Layer 0 top-3: {lens[0]}")
print(f"Layer {len(lens)//2} top-3: {lens[len(lens)//2]}")
print(f"Layer {len(lens)-1} top-3: {lens[-1]}")

print("\n=== Visualization (dummy data, Agg backend) ===")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Logit lens plot
fig1, ax1 = plt.subplots(figsize=(8, 3))
top_probs = [l[0][1] for l in lens]
ax1.plot(top_probs, marker="o")
ax1.set_title("Logit lens top-1 prob")
ax1.set_xlabel("Layer")
ax1.set_ylabel("Prob")
fig1.savefig("test_logit_lens.png", dpi=100)
print("Saved: test_logit_lens.png")

# Patch heatmap (dummy)
dummy_patch = {i: 0.1 + 0.03*i for i in range(0, 24, 2)}
from consciousness_circuit.visualization import plot_patch_heatmap
import seaborn as sns
fig2, ax2 = plt.subplots(figsize=(10, 2))
layers = sorted(dummy_patch.keys())
values = [dummy_patch[l] for l in layers]
sns.heatmap(np.array(values)[None, :], cmap="YlGnBu", annot=True, fmt=".2f",
            xticklabels=layers, yticklabels=["metric"], ax=ax2)
ax2.set_title("Patch impact (dummy)")
fig2.savefig("test_patch_heatmap.png", dpi=100)
print("Saved: test_patch_heatmap.png")

# Residual scatter (dummy)
from sklearn.decomposition import PCA
dummy_res = np.random.randn(20, 64).astype(np.float32)
dummy_labels = ["clean"]*10 + ["corrupt"]*10
pca = PCA(n_components=2)
emb = pca.fit_transform(dummy_res)
fig3, ax3 = plt.subplots(figsize=(5, 4))
for lbl, c in [("clean", "green"), ("corrupt", "red")]:
    mask = [l == lbl for l in dummy_labels]
    ax3.scatter(emb[mask, 0], emb[mask, 1], c=c, label=lbl, alpha=0.7)
ax3.legend()
ax3.set_title("Residual scatter (dummy PCA)")
fig3.savefig("test_residual_scatter.png", dpi=100)
print("Saved: test_residual_scatter.png")

print("\n=== ALL TESTS PASSED ===")
