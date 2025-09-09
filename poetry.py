#!/usr/bin/env python3

import sys
import os
import argparse
import torch

from model import PoetryGPT, save_model_and_tokenizer, load_model_and_tokenizer
from data import create_poetry_dataloader
from training import train_gpt, TrainingConfig


def train_model():
    """Train the model with subword tokenization"""
    print("PoetryGPT Training with Subword Tokenization")
    print("=" * 60)
    
    config = TrainingConfig()
    
    # Use MPS if available, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    config.device = device
    print(f"Training device: {device}")
    print(f"Model configuration: {config}")
    print()

    # Create data loader (this also creates and trains the tokenizer)
    dataloader, dataset = create_poetry_dataloader(
        vocab_size=config.vocab_size,
        seq_length=config.seq_len,
        batch_size=config.batch_size,
        tokenizer_name=config.tokenizer_name
    )
    
    # Get the actual vocabulary size from the trained tokenizer
    actual_vocab_size = dataset.get_vocab_size()
    print(f"\nModel Configuration:")
    print(f"   Vocabulary size: {actual_vocab_size:,} subwords")
    print(f"   Sequence length: {config.seq_len}")
    print(f"   Model dimensions: {config.d_model}")
    print(f"   Attention heads: {config.n_heads}")
    print(f"   Layers: {config.n_layers}")
    print(f"   Feed-forward dim: {config.d_ff}")
    print(f"   Dropout: {config.dropout}")
    print(f"   Activation: {config.activation_type}")

    # Create model with actual vocabulary size
    model = PoetryGPT(
        vocab_size=actual_vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_len=config.max_len,
        dropout=config.dropout,
        init_method=config.init_method,
        activation_type=config.activation_type
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    print(f"\nStarting training...")
    print("=" * 60)
    
    model = train_gpt(model, config)

    print("\n" + "=" * 60)
    print("Training completed!")
    
    # Save model and tokenizer separately
    os.makedirs("static/models", exist_ok=True)
    os.makedirs("static/tokenizer", exist_ok=True)
    save_model_and_tokenizer(model, dataset.tokenizer, "static/models/poetry_gpt.pt", "static/tokenizer/poetry_tokenizer.model")
    
    return model, dataset.tokenizer


def generate_text(prompt="", max_tokens=100):
    """Generate poetry using trained model"""
    print(f"Generating poetry...")
    print(f"   Prompt: '{prompt}'")
    print(f"   Max tokens: {max_tokens}")
    print()
    
    try:
        model, tokenizer = load_model_and_tokenizer("static/models/poetry_gpt.pt", "static/tokenizer/poetry_tokenizer.model")
        model.eval()
        
        # Determine device
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model = model.to(device)
        
        # Encode prompt
        if prompt:
            input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        else:
            # Start with BOS token
            input_ids = torch.tensor([[tokenizer.bos_id]], device=device)
        
        print("Generated poetry:")
        print("-" * 40)
        
        # Generate text
        with torch.no_grad():
            generated = model.generate(
                input_ids, 
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        # Decode and display
        generated_text = tokenizer.decode(generated[0].tolist())
        print(generated_text)
        print("-" * 40)
        
    except FileNotFoundError:
        print(f"Model not found at static/models/poetry_gpt.pt")
        print("   Please train the model first using: python poetry.py train")
        return
    except Exception as e:
        print(f"Error during generation: {e}")
        return


def main():
    parser = argparse.ArgumentParser(description="PoetryGPT - Slovenian Poetry Generator")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate poetry')
    gen_parser.add_argument('--prompt', type=str, default="", help='Starting prompt for generation')
    gen_parser.add_argument('--max-tokens', type=int, default=100, help='Maximum tokens to generate')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model()
    elif args.command == 'generate':
        generate_text(
            prompt=args.prompt,
            max_tokens=args.max_tokens
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()