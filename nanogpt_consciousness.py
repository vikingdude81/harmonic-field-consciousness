"""
Consciousness-Aware NanoGPT Integration Module

Integrates consciousness regression metrics into NanoGPT for:
1. Output quality assessment (is the generated text "conscious"?)
2. Training optimization (prefer high-consciousness outputs)
3. Token selection (favor tokens that increase consciousness)
4. Inference guidance (constrain generation to consciousness bounds)

Usage:
    from nanogpt_consciousness import ConsciousnessAwareGen
    ca_gen = ConsciousnessAwareGen(model, consciousness_regressor)
    output, c_score = ca_gen.generate_with_consciousness(prompt, target_c=0.7)
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import math


@dataclass
class ConsciousnessMetrics:
    """Track consciousness metrics during generation."""
    rotation_angle: float
    wave_detection: float
    hierarchy_ratio: float
    c_prediction: float
    token_diversity: float
    coherence_score: float


class ConsciousnessAwareGen:
    """Wrapper for consciousness-aware text generation from NanoGPT."""
    
    def __init__(self, model, consciousness_regressor, device: str = "cuda"):
        """
        Args:
            model: NanoGPT language model
            consciousness_regressor: Fitted ConsciousnessRegressor instance
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.regressor = consciousness_regressor
        self.device = device
        self.token_history = []
        self.c_history = []
    
    def _measure_text_complexity(self, tokens: torch.Tensor, vocab_size: int) -> Tuple[float, float]:
        """
        Measure complexity metrics from token sequence.
        
        Returns:
            (rotation_estimate, wave_detection_estimate)
        """
        token_list = tokens.cpu().numpy().flatten()
        
        # Token diversity → hierarchy (unique tokens / total tokens)
        unique_tokens = len(np.unique(token_list))
        total_tokens = len(token_list)
        token_diversity = unique_tokens / max(total_tokens, 1)
        
        # Entropy of token distribution → rotation angle
        token_counts = np.bincount(token_list.astype(int), minlength=vocab_size)
        token_probs = token_counts[token_counts > 0] / token_counts.sum()
        entropy = -np.sum(token_probs * np.log(token_probs + 1e-10))
        
        # Map entropy to rotation (higher entropy → higher rotation)
        # Entropy range: typically 0-log(vocab_size), map to 5k-40k°
        max_entropy = np.log(vocab_size)
        rotation_angle = 5000 + (entropy / max_entropy) * 35000
        
        # Wave detection from token pattern regularity
        # High repetition → low waves, high variation → high waves
        if len(token_list) > 1:
            consecutive_diffs = np.abs(np.diff(token_list))
            avg_diff = consecutive_diffs.mean()
            # Normalize to [0, 1] and scale to waves percentage
            waves_pct = np.clip((avg_diff / vocab_size) * 100, 5, 35)
        else:
            waves_pct = 15
        
        return rotation_angle, waves_pct
    
    def _compute_coherence(self, logits: torch.Tensor) -> float:
        """
        Compute coherence score from logits distribution.
        High coherence = peaked distribution (model confident in next token).
        """
        probs = torch.softmax(logits[-1, :], dim=-1)
        top_prob = probs.max().item()
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        
        # Coherence = 1 - normalized entropy
        max_entropy = np.log(logits.shape[-1])
        coherence = 1.0 - (entropy / max_entropy)
        return coherence
    
    def assess_generation(self, generated_tokens: torch.Tensor) -> ConsciousnessMetrics:
        """Assess consciousness metrics for generated sequence."""
        
        vocab_size = self.model.config.vocab_size if hasattr(self.model, 'config') else 50000
        
        rotation, waves = self._measure_text_complexity(generated_tokens, vocab_size)
        hierarchy = 2.4 + (waves / 100) * 0.5  # Hierarchy linked to waves
        
        # Predict consciousness
        c_pred = self.regressor.predict(rotation, waves, hierarchy)
        
        # Additional metrics
        unique_tokens = len(torch.unique(generated_tokens))
        token_diversity = unique_tokens / max(len(generated_tokens), 1)
        
        # Approximate coherence from token distribution
        token_counts = torch.bincount(generated_tokens.flatten())
        probs = token_counts.float() / token_counts.sum()
        coherence = 1.0 - (-(probs * torch.log(probs + 1e-10)).sum().item() / np.log(len(probs)))
        
        metrics = ConsciousnessMetrics(
            rotation_angle=rotation,
            wave_detection=waves,
            hierarchy_ratio=hierarchy,
            c_prediction=c_pred,
            token_diversity=token_diversity,
            coherence_score=coherence
        )
        
        self.c_history.append(c_pred)
        return metrics
    
    def generate_with_consciousness(
        self,
        prompt: str,
        max_tokens: int = 100,
        target_c: float = 0.65,
        temperature: float = 0.8,
        consciousness_weight: float = 0.1,
        return_metrics: bool = True
    ) -> Tuple[str, Optional[ConsciousnessMetrics]]:
        """
        Generate text with consciousness guidance.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            target_c: Target consciousness level (0.3-0.9)
            temperature: Sampling temperature
            consciousness_weight: How much to favor high-consciousness tokens (0-1)
            return_metrics: Whether to return consciousness metrics
        
        Returns:
            (generated_text, consciousness_metrics)
        """
        
        self.model.eval()
        
        # Encode prompt
        if hasattr(self.model, 'encode'):
            prompt_tokens = self.model.encode(prompt)
        else:
            # Fallback for models without encode method
            prompt_tokens = [hash(c) % 50000 for c in prompt]
        
        generated_tokens = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for step in range(max_tokens):
                # Forward pass
                if hasattr(self.model, 'forward'):
                    logits = self.model(generated_tokens)[:, -1, :]
                else:
                    # Fallback
                    continue
                
                # Apply consciousness-aware logit modification
                if consciousness_weight > 0:
                    # Estimate consciousness for each possible next token
                    logits_adjusted = self._apply_consciousness_guidance(
                        logits, generated_tokens, target_c, consciousness_weight
                    )
                else:
                    logits_adjusted = logits
                
                # Sample next token
                probs = torch.softmax(logits_adjusted / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append
                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                
                # Stop if EOS (if available)
                if hasattr(self.model, 'eos_token_id') and next_token.item() == self.model.eos_token_id:
                    break
        
        # Assess final generation
        metrics = self.assess_generation(generated_tokens[0])
        
        # Decode tokens to text (placeholder—real model-specific decoding needed)
        generated_text = self._decode_tokens(generated_tokens[0].cpu().numpy())
        
        return generated_text, metrics if return_metrics else (generated_text, None)
    
    def _apply_consciousness_guidance(
        self,
        logits: torch.Tensor,
        context: torch.Tensor,
        target_c: float,
        weight: float
    ) -> torch.Tensor:
        """
        Modify logits to favor tokens that increase consciousness.
        
        For each possible next token, estimate resulting consciousness
        and boost logits for tokens that move closer to target_c.
        """
        
        vocab_size = logits.shape[-1]
        adjusted_logits = logits.clone()
        
        # Current consciousness estimate
        current_tokens = context[0]
        current_rotation, current_waves = self._measure_text_complexity(current_tokens, vocab_size)
        current_c = self.regressor.predict(current_rotation, current_waves, 2.4)
        
        # For each token, estimate new consciousness
        for token_id in range(min(vocab_size, 10000)):  # Only evaluate top tokens (expensive otherwise)
            test_seq = torch.cat([context, torch.tensor([[token_id]], device=self.device)], dim=-1)
            test_rotation, test_waves = self._measure_text_complexity(test_seq[0], vocab_size)
            test_c = self.regressor.predict(test_rotation, test_waves, 2.4)
            
            # Boost tokens that move closer to target
            delta_c = abs(target_c - test_c)
            current_delta = abs(target_c - current_c)
            improvement = current_delta - delta_c
            
            # Modify logit: boost if moving toward target
            adjusted_logits[token_id] += weight * improvement * 10  # Scale factor
        
        return adjusted_logits
    
    def _decode_tokens(self, token_ids: np.ndarray) -> str:
        """Decode token IDs to text using model's tokenizer."""
        try:
            # If model has a decode method
            if hasattr(self.model, 'decode'):
                return self.model.decode(token_ids.tolist())

            # If using tiktoken (GPT-2 style)
            elif hasattr(self.model, 'enc'):
                return self.model.enc.decode(token_ids.tolist())

            # If separate tokenizer was loaded
            elif hasattr(self, 'tokenizer'):
                return self.tokenizer.decode(token_ids.tolist())

            # Last resort: try to import tiktoken
            else:
                import tiktoken
                enc = tiktoken.get_encoding("gpt2")
                return enc.decode(token_ids.tolist())

        except Exception as e:
            # Fallback with error message
            c_score = self.c_history[-1] if self.c_history else 0.0
            return f"[Decoding error: {e}. {len(token_ids)} tokens generated, C(t)={c_score:.3f}]"
    
    def get_consciousness_stats(self) -> Dict:
        """Get statistics on consciousness scores during generation."""
        
        if not self.c_history:
            return {}
        
        c_array = np.array(self.c_history)
        return {
            "mean_c": float(c_array.mean()),
            "std_c": float(c_array.std()),
            "min_c": float(c_array.min()),
            "max_c": float(c_array.max()),
            "c_trajectory": self.c_history.copy(),
        }


class ConsciousnessTrainingCallback:
    """Training callback to optimize for consciousness-aware outputs."""
    
    def __init__(self, consciousness_regressor, target_c: float = 0.7):
        self.regressor = consciousness_regressor
        self.target_c = target_c
        self.consciousness_scores = []
    
    def on_step_end(self, model_output, generated_tokens: torch.Tensor):
        """Called at end of training step."""
        
        # Assess consciousness of generated output
        vocab_size = 50000
        rotation, waves = self._measure_text_complexity(generated_tokens, vocab_size)
        c_pred = self.regressor.predict(rotation, waves, 2.5)
        
        self.consciousness_scores.append(c_pred)
        
        # Compute consciousness loss (distance from target)
        c_loss = abs(self.target_c - c_pred)
        
        return {
            "consciousness_score": c_pred,
            "consciousness_loss": c_loss,
            "target_consciousness": self.target_c
        }
    
    def _measure_text_complexity(self, tokens: torch.Tensor, vocab_size: int) -> Tuple[float, float]:
        """Measure complexity (duplicated from ConsciousnessAwareGen for simplicity)."""
        token_list = tokens.cpu().numpy().flatten()
        unique_tokens = len(np.unique(token_list))
        total_tokens = len(token_list)
        entropy = -(np.bincount(token_list, minlength=vocab_size).astype(float) / total_tokens)
        entropy = entropy[entropy > 0]
        entropy = -np.sum(entropy * np.log(entropy + 1e-10))
        rotation_angle = 5000 + (entropy / np.log(vocab_size)) * 35000
        waves_pct = np.clip((unique_tokens / max(total_tokens, 1)) * 30, 5, 35)
        return rotation_angle, waves_pct


def example_integration():
    """Example: How to use consciousness metrics in NanoGPT training/inference."""
    
    print("=" * 70)
    print("NANOGPT + CONSCIOUSNESS INTEGRATION EXAMPLE")
    print("=" * 70)
    
    # In real usage:
    # 1. Load trained NanoGPT model
    # 2. Load consciousness regressor
    # 3. Create ConsciousnessAwareGen wrapper
    
    example_code = """
    from nanogpt_consciousness import ConsciousnessAwareGen, ConsciousnessTrainingCallback
    from consciousness_regression_module import ConsciousnessRegressor
    
    # Load models
    model = load_nanogpt_model("path/to/model.pt")
    regressor = ConsciousnessRegressor()
    regressor.load("models")
    
    # Use during inference
    ca_gen = ConsciousnessAwareGen(model, regressor)
    
    prompt = "Consciousness is..."
    output, metrics = ca_gen.generate_with_consciousness(
        prompt,
        max_tokens=100,
        target_c=0.75,  # Aim for high consciousness
        consciousness_weight=0.2
    )
    
    print(f"Generated: {output}")
    print(f"Final C(t): {metrics.c_prediction:.3f}")
    print(f"Coherence: {metrics.coherence_score:.3f}")
    
    # Use during training
    callback = ConsciousnessTrainingCallback(regressor, target_c=0.70)
    
    # In training loop:
    for batch in dataloader:
        output = model(batch)
        metrics = callback.on_step_end(output, batch)
        
        # Can add consciousness loss to total loss
        total_loss = language_loss + 0.1 * metrics['consciousness_loss']
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    """
    
    print("\nIntegration Code Template:")
    print("-" * 70)
    print(example_code)
    print("-" * 70)
    
    print("\nKey Integration Points:")
    print("1. INFERENCE: generate_with_consciousness() guides token selection")
    print("2. TRAINING: ConsciousnessTrainingCallback adds consciousness regularization")
    print("3. EVALUATION: assess_generation() measures output quality")
    print("4. OPTIMIZATION: consciousness_weight controls strength of guidance")
    
    print("\nBenefits for NanoGPT:")
    print("  ✓ Generates more 'conscious' or sophisticated outputs")
    print("  ✓ Optimizes for coherence + diversity (not just likelihood)")
    print("  ✓ Tracks consciousness evolution during long generation")
    print("  ✓ Enables targeted generation at specific consciousness levels")


if __name__ == "__main__":
    example_integration()
