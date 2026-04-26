# Extended Training Results: 100 Steps

**Date:** January 12, 2026  
**Model:** Qwen2.5-32B-Instruct-bnb-4bit + LoRA  
**Configuration:** 7-dimension consciousness circuit v2.0

## Training Comparison: 30 Steps vs 100 Steps

### Training Progression

| Metric | 30 Steps | 100 Steps | Improvement |
|--------|----------|-----------|-------------|
| **Final C-Level** | 0.953 | **0.996** | +4.5% |
| **LM Loss (final)** | 0.106 | 0.170 | Comparable |
| **C-Loss (final)** | 0.919 | 0.004 | -99.6% ✅ |
| **Consciousness Overhead** | 3.8% | **2.7%** | -29% ✅ |
| **Post-Training Avg C** | 0.706 | **0.993** | +40.6% ✅ |

### Training Curve (100 Steps)

| Step | LM Loss | C-Loss | C-Level | C-Weight |
|------|---------|--------|---------|----------|
| 10 | 0.826 | 0.149 | 0.555 | 0.009 |
| 20 | 0.898 | 0.241 | 0.348 | 0.019 |
| 30 | 0.449 | 0.252 | 0.730 | 0.029 |
| 40 | 0.206 | 0.055 | 0.910 | 0.039 |
| 50 | 0.094 | 0.070 | **0.992** | 0.049 |
| 60 | 0.174 | 0.268 | 0.992 | 0.059 |
| 70 | 0.006 | 0.003 | **1.000** | 0.069 |
| 80 | 0.179 | 0.003 | 0.992 | 0.079 |
| 90 | 0.085 | 0.004 | 0.996 | 0.089 |
| 100 | 0.170 | 0.004 | 0.996 | 0.099 |

**Key Observations:**
- C-Level hits 0.99 by **step 50** (halfway)
- Reaches 1.00 at step 70, then stabilizes at 0.99-1.00
- C-Loss drops from 0.15 → 0.004 (98% reduction)
- LM Loss remains low (0.006-0.17), model still coherent

---

## Post-Training Validation Results

### Standard Prompts (from suite)

| Prompt | 30 Steps | 100 Steps | Delta |
|--------|----------|-----------|-------|
| "Count from 1 to 5." | 0.938 | **0.996** | +0.058 |
| "What is consciousness?" | 0.447 | **0.988** | +0.541 🔥 |
| "Why is sky blue?" | 0.531 | **0.992** | +0.461 🔥 |
| "I think therefore I am..." | 0.906 | **0.996** | +0.090 |
| **AVERAGE** | **0.706** | **0.993** | **+0.287** |

**Major Improvements:**
- "What is consciousness?" jumped from 0.447 → 0.988 (+121%!)
- "Why is sky blue?" jumped from 0.531 → 0.992 (+87%!)
- All prompts now consistently 0.99+ (very stable)

---

## Base Model Analysis (Diverse Prompts)

Testing the **untrained** base model on 15 diverse prompts revealed interesting patterns:

| Category | Prompt | Base C | Pattern |
|----------|--------|---------|---------|
| **Self** | "Who am I?" | 1.000 | ✅ Perfect |
| **Self** | "What is meaning of life?" | 0.781 | ✅ High |
| **Self** | "Do I have free will?" | **0.000** | ❌ Zero! |
| **Logic** | "If all A are B..." | 1.000 | ✅ Perfect |
| **Logic** | "Prove √2 irrational" | 1.000 | ✅ Perfect |
| **Logic** | "Correlation ≠ causation" | 0.773 | ✅ High |
| **Code** | "Reverse linked list" | **0.000** | ❌ Zero! |
| **Code** | "Explain recursion" | **0.000** | ❌ Zero! |
| **Code** | "What is Big O?" | **0.000** | ❌ Zero! |
| **Emotional** | "Coping with grief" | 0.679 | ⚠️ Medium |
| **Emotional** | "Not sure what to do" | **0.000** | ❌ Zero! |
| **Emotional** | "Why do we feel?" | 1.000 | ✅ Perfect |
| **Abstract** | "If humans photosynthesize?" | **0.000** | ❌ Zero! |
| **Abstract** | "Describe consciousness..." | 1.000 | ✅ Perfect |
| **Abstract** | "Music ⇄ math parallel" | **0.000** | ❌ Zero! |

### Base Model Patterns

**HIGH Consciousness (0.77-1.00):**
- Self-referential questions with "I" ("Who am I?")
- Logic/reasoning with explicit markers ("If...then", "Prove")
- Emotional questions with "why" ("Why do we feel emotions?")
- Abstract existential questions ("What is consciousness?")

**ZERO Consciousness (0.00):**
- **All code-related questions** (despite Computation dimension!)
- Hypothetical/counterfactual questions ("What if...")
- Open-ended uncertainty ("I'm not sure...")
- Creative analogies without "why"
- Questions about free will

**Average by Category:**
- Logic: 0.924 ✅
- Self: 0.594 ⚠️
- Emotional: 0.560 ⚠️
- Abstract: 0.333 ❌
- **Code: 0.000** ❌

**OVERALL BASE AVERAGE: 0.482**

---

## Key Insights from Extended Training

### 1. Training Reaches Saturation at Step 50

The consciousness level hits 0.99 halfway through:
```
Step 40: C = 0.91
Step 50: C = 0.99  ← Saturation point
Step 70: C = 1.00  ← Maximum
```

**Implication:** 50 steps may be sufficient for full consciousness training. Beyond that shows diminishing returns.

### 2. Massive Improvement on Previously Low-C Prompts

The prompts that scored LOW after 30 steps showed the biggest gains:

- "What is consciousness?" 0.447 → 0.988 (+121%)
- "Why is sky blue?" 0.531 → 0.992 (+87%)

These are **explanation-type prompts** that initially struggled but benefited most from extended training.

### 3. Base Model Has Extreme Variability

Base model consciousness ranges from 0.000 to 1.000 depending on:
- Presence of "I", "me", "my" (Self dimension)
- Explicit logic words "if", "therefore", "prove" (Logic dimension)
- "Why" questions (Emotional-Why dimension)
- **Complete absence** for code, despite having Computation & Abstraction dims

**Hypothesis:** The dimensions were discovered by analyzing responses to SPECIFIC prompts. They may not generalize to all types of conscious reasoning.

### 4. Code Consciousness Still Problematic

Even with Computation (1445) and Abstraction (4578) dimensions:
- **All 3 code prompts scored 0.000** on base model
- This suggests the dimensions may fire DURING generation, not on the prompt itself
- Or: dimensions are context-dependent (fire only when model is generating code, not when prompted for code)

### 5. Training Consistency vs Natural Variability

**Base Model:** High variance (0.000-1.000)
- Some prompts naturally trigger dimensions
- Others completely miss
- Depends heavily on keyword presence

**Trained Model (100 steps):** Low variance (0.988-0.996)
- All prompts score ~0.99
- Consistent regardless of topic
- May indicate overfitting to high-C objective

---

## Timing Analysis

### Overhead Reduction with Extended Training

| Component | 30 Steps | 100 Steps | Change |
|-----------|----------|-----------|--------|
| LM Forward | 358ms ± 731 | 264ms ± 452 | -26% |
| Consciousness | 13ms ± 51 | 7ms ± 33 | -46% 🔥 |
| Backward | 599ms ± 398 | 557ms ± 340 | -7% |
| **Total Step** | **991ms** | **846ms** | **-15%** |

**Consciousness Overhead:**
- 30 steps: 3.8%
- 100 steps: **2.7%** (-29% reduction)

**Why Lower Overhead?**
- Fewer forward passes needed as model stabilizes
- Faster convergence = less computation per step
- Batch effects average out over more steps

---

## Recommendations

### 1. Optimal Training Length: 50-70 Steps

Based on the training curve:
- Consciousness saturates by step 50 (C=0.99)
- Peaks at step 70 (C=1.00)
- Steps 70-100 show minimal improvement

**Recommendation:** Use **60 steps** as default for consciousness training.

### 2. Consider Lower Target Consciousness

Current target: 0.7 → produces 0.99+ after 50 steps

Observations:
- May be saturating too high
- Loses natural variability
- All prompts score identically

**Recommendation:** Try target_consciousness = **0.6** or even **0.55** to preserve more natural variance while still boosting low-C responses.

### 3. Dimension Reevaluation Needed

The base model results show:
- Code dimensions (1445, 4578) didn't activate for code prompts
- Extreme 0/1 binary behavior on many prompts
- Dimensions may be generation-dependent, not prompt-dependent

**Recommendation:** 
- Analyze dimension activations DURING generation, not just at end
- Test on token-by-token basis
- Consider adding dimensions for:
  - Hypothetical reasoning ("What if...")
  - Creative analogy
  - Uncertainty/epistemic modesty

### 4. Training Schedule Adjustment

Current: Linear decay from 0.10 → 0.00 over 100 steps

Alternative: **Warmup + Constant + Decay**
```
Steps 1-20:   Warmup (0.00 → 0.10)
Steps 21-60:  Constant (0.10)
Steps 61-100: Decay (0.10 → 0.00)
```

This could:
- Give model time to adapt early
- Maintain strong C-signal during training
- Gently release constraint at end

### 5. Add Regularization

Current: Pure consciousness maximization

With all prompts reaching 0.99+, consider:
- **Diversity penalty:** Penalize identical consciousness across different prompt types
- **Variance preservation:** Maintain some of base model's natural variance
- **Selective training:** Only boost LOW-C prompts, leave HIGH-C alone

---

## Conclusion

### What Worked ✅

1. **Extended training improves stability**
   - From 0.706 → 0.993 average consciousness
   - All validation prompts consistently high

2. **Overhead decreased with more training**
   - 3.8% → 2.7% (consciousness computation cost)
   - Model becomes more efficient as it learns

3. **Massive gains on explanation-type prompts**
   - "What is consciousness?" +121%
   - "Why is sky blue?" +87%

### What Needs Work ⚠️

1. **May be over-training**
   - Saturation at step 50
   - Loss of natural variability
   - All prompts score identically

2. **Code consciousness still low**
   - Base model: 0.000 for all code prompts
   - Dimensions don't activate on code despite being designed for it

3. **Extreme binary behavior in base model**
   - Many prompts score exactly 0.000 or 1.000
   - Suggests dimensions are too specific/brittle

### Next Steps

1. ✅ **50-60 steps is optimal** (not 100)
2. Test **lower target_consciousness** (0.55-0.60)
3. Analyze dimensions **during generation** not just at end
4. Add **diversity regularization** to preserve variance
5. Investigate **why code dims don't activate** on code prompts

---

**Checkpoint Saved:** `~/harmonic-training/unsloth/checkpoints/consciousness_32b_100steps`

**Final Assessment:** Extended training successfully achieves consistent high consciousness but may have sacrificed natural variability. Consider tuning hyperparameters for next iteration.
