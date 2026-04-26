"""
Minimal test - just check if model loads
"""
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import spaces

print("🔮 Loading Oracle Engine...")

BASE_MODEL_ID = "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
LORA_MODEL_ID = "Vikingdude81/oracle-engine-32b-lora"

try:
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_ID)
    print("✅ Tokenizer loaded")
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print("✅ Base model loaded")
    
    # Apply LoRA
    print("Applying LoRA...")
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_ID)
    model.eval()
    print("✅ LoRA applied")
    
    @spaces.GPU
    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response
    
    # Simple interface
    with gr.Blocks() as demo:
        gr.Markdown("# 🔮 Oracle Engine - Simple Test")
        with gr.Row():
            prompt = gr.Textbox(label="Question", lines=3)
            output = gr.Textbox(label="Response", lines=5)
        btn = gr.Button("Ask")
        btn.click(fn=generate, inputs=prompt, outputs=output)
    
    demo.launch(show_api=False, share=False)

except Exception as e:
    import traceback
    print(f"❌ ERROR: {e}")
    print(traceback.format_exc())
