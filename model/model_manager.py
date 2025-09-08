import os
import torch
import pickle
from pathlib import Path


def save_model_and_tokenizer(model, tokenizer, config, path="checkpoints/model_complete.pt"):
    """Save model, tokenizer, and config together"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'config': config,
        'vocab_size': tokenizer.vocab_size,
    }
    
    torch.save(save_dict, path)
    print(f"Model and tokenizer saved to: {path}")


def load_model_and_tokenizer(path="checkpoints/model_complete.pt"):
    """Load model, tokenizer, and config"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # Import here to avoid circular imports
    from model import PoetryGPT
    
    # Load with weights_only=False for custom objects like tokenizer
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    tokenizer = checkpoint['tokenizer']
    config = checkpoint['config']
    
    model = PoetryGPT(
        vocab_size=checkpoint['vocab_size'],
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_len=config.max_len,
        dropout=config.dropout
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model and tokenizer loaded from: {path}")
    
    return model, tokenizer, config