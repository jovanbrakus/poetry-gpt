#!/usr/bin/env python3

import sys
import argparse
import torch

from model import MiniGPT, save_model_and_tokenizer, load_model_and_tokenizer
from data import CharTokenizer, load_slovenian_data
from training import train_gpt, TrainingConfig


def train_model():
    """Train the model and save it locally"""
    print("Loading Slovenian poetry data...")
    text = load_slovenian_data("static/slovenian")
    print(f"Dataset size: {len(text):,} characters")
    print(f"First 200 characters:\n{text[:200]}")
    print("\n" + "="*50 + "\n")

    tokenizer = CharTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Sample vocabulary: {list(tokenizer.char_to_idx.items())[:10]}")

    config = TrainingConfig()
    
    # Use MPS if available, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    config.device = device
    print(f"Training on: {device}")

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_len=config.max_len,
        dropout=config.dropout
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    print("\nStarting training...")
    model = train_gpt(model, text, tokenizer, config)

    # Save the complete model
    save_model_and_tokenizer(model, tokenizer, config)
    
    print("\nTraining complete! Model saved locally.")


def generate_text(prompt):
    """Load the model and generate text from the given prompt"""
    try:
        model, tokenizer, config = load_model_and_tokenizer()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using: python poetry.py train")
        return

    # Use MPS if available, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    model.eval()

    print(f"Using device: {device}")
    print(f"Generating text from prompt: '{prompt}'")
    print("-" * 50)

    # Encode the prompt
    try:
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    except KeyError as e:
        print(f"Error: Character '{e.args[0]}' not found in vocabulary.")
        print("The model was trained on Slovenian text. Please use characters that exist in the training data.")
        return

    # Generate text
    with torch.no_grad():
        generated = model.generate(
            context, 
            max_new_tokens=200, 
            temperature=0.8, 
            top_k=50
        )

    # Decode and display
    full_text = tokenizer.decode(generated[0].tolist())
    print(full_text)


def main():
    parser = argparse.ArgumentParser(description='Slovenian Poetry Generator')
    parser.add_argument('command', choices=['train', 'generate'], 
                       help='Command to run: train the model or generate text')
    parser.add_argument('prompt', nargs='?', default='', 
                       help='Text prompt for generation (required for generate command)')

    args = parser.parse_args()

    if args.command == 'train':
        train_model()
    elif args.command == 'generate':
        if not args.prompt:
            print("Error: Please provide a prompt for text generation.")
            print("Usage: python poetry.py generate \"Your prompt here\"")
            sys.exit(1)
        generate_text(args.prompt)


if __name__ == "__main__":
    main()