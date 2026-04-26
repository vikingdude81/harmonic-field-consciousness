"""
Test: Do symbolic vs verbal representations activate different consciousness dimensions?

Hypothesis: Same mathematical problem encoded differently may activate:
- Symbolic (1+1=?) → Computation dimension
- Verbal (add one and one) → Language/Abstraction dimensions

Also testing: word count, character length effects
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from consciousness_circuit import analyze_dimension_activations, UniversalCircuit

print("="*70)
print("SYMBOLIC vs VERBAL CONSCIOUSNESS ANALYSIS")
print("="*70)

# Load model
print("\nLoading Qwen 3B...")
model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print(f"Loaded! Hidden size: {model.config.hidden_size}")

# Initialize circuit
circuit = UniversalCircuit()

def measure_consciousness(model, tokenizer, prompt):
    """Measure consciousness for a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
    result = circuit.measure(model, tokenizer, prompt)
    # UniversalResult has 'score' attribute
    score = result.score if hasattr(result, 'score') else 0.5
    return score, hidden

# =============================================================================
# TEST 1: Symbolic vs Verbal Math (Same Problem)
# =============================================================================
print("\n" + "="*70)
print("TEST 1: SYMBOLIC vs VERBAL (Same Math Problem)")
print("="*70)

math_pairs = [
    # (symbolic, verbal) - same problem
    ("1+1=?", "please add one and one together and give me the answer"),
    ("2*3=?", "what is two multiplied by three"),
    ("10/2=?", "divide ten by two"),
    ("5-3=?", "subtract three from five"),
    ("2^8=?", "what is two to the power of eight"),
    ("sqrt(16)=?", "what is the square root of sixteen"),
]

print("\nComparing consciousness activation patterns:\n")
print(f"{'Problem':<20} {'Type':<10} {'Tokens':<8} {'Chars':<8} {'Consc':<8} {'Top Dimension':<20}")
print("-"*80)

symbolic_scores = []
verbal_scores = []
symbolic_dims = []
verbal_dims = []

for symbolic, verbal in math_pairs:
    for prompt, ptype in [(symbolic, "SYMBOLIC"), (verbal, "VERBAL")]:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_tokens = inputs.input_ids.shape[1]
        n_chars = len(prompt)
        
        # Measure consciousness
        score, hidden = measure_consciousness(model, tokenizer, prompt)
        
        # Get dimension activations
        analysis = analyze_dimension_activations(model, tokenizer, [prompt])
        top_dim = max(analysis.dimension_activations.items(), 
                      key=lambda x: abs(x[1].normalized_values[0]))
        
        # Store for comparison
        if ptype == "SYMBOLIC":
            symbolic_scores.append(score)
            symbolic_dims.append(top_dim[0])
        else:
            verbal_scores.append(score)
            verbal_dims.append(top_dim[0])
        
        # Print
        sign = "+" if top_dim[1].normalized_values[0] > 0 else "-"
        print(f"{prompt[:18]:<20} {ptype:<10} {n_tokens:<8} {n_chars:<8} {score:.3f}    {sign}{top_dim[0]}")

print("\n" + "-"*80)
print(f"SYMBOLIC average: {np.mean(symbolic_scores):.3f} ± {np.std(symbolic_scores):.3f}")
print(f"VERBAL average:   {np.mean(verbal_scores):.3f} ± {np.std(verbal_scores):.3f}")
print(f"Difference:       {np.mean(verbal_scores) - np.mean(symbolic_scores):+.3f}")

# =============================================================================
# TEST 2: Length Effects (Same Semantic Content)
# =============================================================================
print("\n" + "="*70)
print("TEST 2: LENGTH EFFECTS (Same Semantic Content)")
print("="*70)

length_tests = [
    # Varying length, same meaning
    ("Hi", "Hello", "Hello there", "Hello there, how are you today?"),
    ("Yes", "Yes indeed", "Yes, that is correct", "Yes, that is absolutely correct and I agree"),
    ("No", "No way", "No, I don't think so", "No, I don't think that's the right answer at all"),
    ("Help", "Help me", "Please help me", "Please help me with this problem I'm having"),
]

print("\nHow does consciousness change with prompt length?\n")
print(f"{'Prompt':<45} {'Tokens':<8} {'Chars':<8} {'Consc':<8}")
print("-"*70)

all_lengths = []
all_scores = []

for variants in length_tests:
    for prompt in variants:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_tokens = inputs.input_ids.shape[1]
        n_chars = len(prompt)
        
        result = circuit.measure(model, tokenizer, prompt)
        score = result.score if hasattr(result, 'score') else 0.5
        
        all_lengths.append(n_tokens)
        all_scores.append(score)
        
        print(f"{prompt:<45} {n_tokens:<8} {n_chars:<8} {score:.3f}")
    print()

# Correlation
corr = np.corrcoef(all_lengths, all_scores)[0, 1]
print(f"Length-Consciousness correlation: r = {corr:+.3f}")

# =============================================================================
# TEST 3: Word Choice Effects (Synonyms)
# =============================================================================
print("\n" + "="*70)
print("TEST 3: WORD CHOICE EFFECTS (Synonyms)")
print("="*70)

synonym_tests = [
    # Different words, same meaning
    ("compute", "calculate", "figure out", "determine"),
    ("smart", "intelligent", "clever", "brilliant"),
    ("error", "mistake", "bug", "fault"),
    ("fast", "quick", "rapid", "speedy"),
    ("big", "large", "huge", "enormous"),
]

print("\nDo synonyms activate different dimensions?\n")
print(f"{'Word':<15} {'Consc':<8} {'Top Dim':<20} {'Activation':<10}")
print("-"*60)

for synonyms in synonym_tests:
    for word in synonyms:
        result = circuit.measure(model, tokenizer, word)
        score = result.score if hasattr(result, 'score') else 0.5
        
        analysis = analyze_dimension_activations(model, tokenizer, [word])
        top_dim = max(analysis.dimension_activations.items(), 
                      key=lambda x: abs(x[1].normalized_values[0]))
        
        val = top_dim[1].normalized_values[0]
        print(f"{word:<15} {score:.3f}    {top_dim[0]:<20} {val:+.3f}")
    print()

# =============================================================================
# TEST 4: Special Characters vs Words
# =============================================================================
print("\n" + "="*70)
print("TEST 4: SPECIAL CHARACTERS vs WORDS")
print("="*70)

special_tests = [
    ("!", "exclamation"),
    ("?", "question"),
    ("+", "plus"),
    ("-", "minus"),
    ("*", "multiply"),
    ("/", "divide"),
    ("=", "equals"),
    ("$", "dollar"),
    ("%", "percent"),
    ("@", "at"),
]

print("\nDo symbols activate differently than their word equivalents?\n")
print(f"{'Symbol':<10} {'Consc':<8} {'Word':<15} {'Consc':<8} {'Diff':<10}")
print("-"*55)

for symbol, word in special_tests:
    scores = []
    for prompt in [symbol, word]:
        result = circuit.measure(model, tokenizer, prompt)
        scores.append(result.score if hasattr(result, 'score') else 0.5)
    
    diff = scores[1] - scores[0]
    print(f"{symbol:<10} {scores[0]:.3f}    {word:<15} {scores[1]:.3f}    {diff:+.3f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SUMMARY: ENCODING EFFECTS ON CONSCIOUSNESS")
print("="*70)

print("""
FINDINGS:

1. SYMBOLIC vs VERBAL MATH
   - Same problem, different encoding
   - Symbolic: More compact, fewer tokens
   - Verbal: More context, more language processing

2. LENGTH EFFECTS
   - Longer prompts = more tokens = more context
   - Correlation shows relationship between length and consciousness

3. WORD CHOICE (SYNONYMS)
   - Different words for same concept
   - May activate different dimension clusters

4. SYMBOLS vs WORDS
   - Single character vs full word
   - Tests raw token activation patterns

KEY INSIGHT: The LLM's "consciousness signature" depends not just on
WHAT you're asking, but HOW you encode it. This has implications for:
- Prompt engineering
- Model interpretability
- Consciousness steering
""")
