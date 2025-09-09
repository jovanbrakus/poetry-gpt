import os
import torch
import pickle
from pathlib import Path


def save_model_only(model, path="static/models/poetry_gpt.pt", config=None):
    """Save model and config only (no tokenizer)"""
    dir_path = os.path.dirname(path)
    if dir_path:  # Only create directory if path has a directory component
        os.makedirs(dir_path, exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    if config is not None:
        save_dict['config'] = config
    
    torch.save(save_dict, path)
    print(f"Model saved to: {path}")


def save_tokenizer(tokenizer, path="static/tokenizer/poetry_tokenizer.model"):
    """Save tokenizer separately"""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    tokenizer.save(path)
    print(f"Tokenizer saved to: {path}")


def save_model_and_tokenizer(model, tokenizer, model_path="static/models/poetry_gpt.pt", tokenizer_path="static/tokenizer/poetry_tokenizer.model", config=None):
    """Save model and tokenizer separately"""
    save_model_only(model, model_path, config)
    save_tokenizer(tokenizer, tokenizer_path)


def load_tokenizer(tokenizer_path="static/tokenizer/poetry_tokenizer.model"):
    """Load tokenizer from separate file"""
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    
    # Import here to avoid circular imports
    from data.tokenizer import SubwordTokenizer
    
    tokenizer = SubwordTokenizer()
    tokenizer.load(tokenizer_path)
    
    print(f"Tokenizer loaded from: {tokenizer_path}")
    return tokenizer


def load_model_only(model_path="static/models/poetry_gpt.pt", vocab_size=None):
    """Load model only (no tokenizer)"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Import here to avoid circular imports
    from model import PoetryGPT
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    
    # Try to get config, otherwise infer from state dict
    if 'config' in checkpoint and hasattr(checkpoint['config'], 'd_model'):
        config = checkpoint['config']
        model_vocab_size = vocab_size or getattr(config, 'vocab_size', None)
        
        if model_vocab_size is None:
            # Infer vocab size from embedding layer
            model_vocab_size = checkpoint['model_state_dict']['token_embedding.weight'].shape[0]
        
        model = PoetryGPT(
            vocab_size=model_vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_len=config.max_len,
            dropout=config.dropout,
            activation_type=getattr(config, 'activation_type', 'swiglu')
        )
    else:
        # Infer model architecture from state dict
        state_dict = checkpoint['model_state_dict']
        model_vocab_size = vocab_size or state_dict['token_embedding.weight'].shape[0]
        d_model = state_dict['token_embedding.weight'].shape[1]
        max_len = state_dict['position_encoding.pe'].shape[1] if 'position_encoding.pe' in state_dict else 2048
        
        # Count layers by finding max block number
        layer_numbers = [int(key.split('.')[1]) for key in state_dict.keys() if key.startswith('blocks.')]
        n_layers = max(layer_numbers) + 1 if layer_numbers else 0
        
        # Infer feed-forward dimension from first layer
        ff_key = 'blocks.0.feed_forward.gate_proj.weight'
        if ff_key in state_dict:
            d_ff = state_dict[ff_key].shape[0]
        else:
            d_ff = d_model * 4  # Default
        
        # Create model with inferred parameters
        model = PoetryGPT(
            vocab_size=model_vocab_size,
            d_model=d_model,
            n_heads=8,  # Default
            n_layers=n_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=0.1,  # Default
            activation_type='swiglu'
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from: {model_path}")
    return model


def load_model_and_tokenizer(model_path="static/models/poetry_gpt.pt", tokenizer_path="static/tokenizer/poetry_tokenizer.model"):
    """Load model and tokenizer from separate files"""
    tokenizer = load_tokenizer(tokenizer_path)
    model = load_model_only(model_path, vocab_size=tokenizer.get_vocab_size())
    
    return model, tokenizer