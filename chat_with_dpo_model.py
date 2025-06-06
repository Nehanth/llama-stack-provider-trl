#!/usr/bin/env python3
"""
Simple CLI chat interface for the trained DPO model.
Usage: python chat_with_dpo_model.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

def load_model(model_path="./checkpoints/dpo_model"):
    """Load the trained DPO model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        print(f"Model loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model path is correct and the model was saved properly.")
        sys.exit(1)

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7, top_p=0.9):
    """Generate a response from the model."""
    
    # Format the prompt (you can customize this based on your training format)
    formatted_prompt = f"Human: {prompt}\n\nAssistant:"
    
    # Tokenize input
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    
    # Move to same device as model
    inputs = inputs.to(next(model.parameters()).device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "Assistant:" in full_response:
        response = full_response.split("Assistant:")[-1].strip()
    else:
        response = full_response[len(formatted_prompt):].strip()
    
    return response

def main():
    print("DPO Model Chat Interface")
    print("=" * 50)
    
    # Load model
    model, tokenizer = load_model()
    
    print("\nReady to chat! Type 'quit', 'exit', or 'q' to end the conversation.")
    print("Type 'help' for usage tips.")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            elif user_input.lower() == 'help':
                print("\nUsage Tips:")
                print("- Ask questions or give instructions")
                print("- The model was trained with DPO (Direct Preference Optimization)")
                print("- Type 'quit', 'exit', or 'q' to end")
                print("- Type 'clear' to clear the screen")
                continue
            elif user_input.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            elif not user_input:
                print("Please enter a message or type 'help' for usage tips.")
                continue
            
            # Generate response
            print("\nAssistant: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError generating response: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main() 