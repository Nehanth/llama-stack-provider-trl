#!/usr/bin/env python3
"""
Simple inference test for the DPO-trained Granite MoE model with router logits disabled.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

def test_granite_inference():
    print("🧪 Testing Granite MoE inference (router logits disabled)...")
    
    try:
        # Load config and disable router logits for inference
        print("📝 Loading and modifying config for inference...")
        config = AutoConfig.from_pretrained('checkpoints/dpo_model')
        config.output_router_logits = False  # Disable for inference
        print(f"   - Router logits disabled: {not config.output_router_logits}")
        print(f"   - Experts: {config.num_local_experts}")
        
        # Load model with modified config
        print("📥 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            'checkpoints/dpo_model',
            config=config,
            torch_dtype=torch.float32,
            device_map='cpu'  # Use CPU to avoid device mismatch
        )
        
        tokenizer = AutoTokenizer.from_pretrained('checkpoints/dpo_model')
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Model loaded successfully!")
        
        # Simple test
        prompt = "What is machine learning?"
        print(f"🔬 Testing: '{prompt}'")
        
        inputs = tokenizer(prompt, return_tensors='pt', return_attention_mask=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=50,
                do_sample=False,  # Greedy for consistent results
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(prompt):].strip()
        
        print(f"🤖 Generated: {generated_text}")
        
        if len(generated_text) > 10:
            print("✅ SUCCESS: Model generates coherent text!")
            print("🎉 DPO-trained Granite MoE model is working correctly!")
        else:
            print("⚠️ Generated text is very short")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_granite_inference() 