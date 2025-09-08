import os
import torch
import pickle
from pathlib import Path


def save_model_and_tokenizer(model, tokenizer, path="static/checkpoints/model_complete.pt", config=None):
    """Save model, tokenizer, and config together"""
    dir_path = os.path.dirname(path)
    if dir_path:  # Only create directory if path has a directory component
        os.makedirs(dir_path, exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'vocab_size': tokenizer.get_vocab_size(),
    }
    
    if config is not None:
        save_dict['config'] = config
    
    torch.save(save_dict, path)
    print(f"Model and tokenizer saved to: {path}")


def load_model_and_tokenizer(path="static/checkpoints/model_complete.pt"):
    """Load model and tokenizer"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # Import here to avoid circular imports
    from model import PoetryGPT
    
    # Load with weights_only=False for custom objects like tokenizer
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    tokenizer = checkpoint['tokenizer']
    vocab_size = checkpoint['vocab_size']
    
    # Try to get config, otherwise use defaults from the saved model state
    if 'config' in checkpoint and hasattr(checkpoint['config'], 'd_model'):
        config = checkpoint['config']
        model = PoetryGPT(
            vocab_size=vocab_size,
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
        d_model = state_dict['token_embedding.weight'].shape[1]
        max_len = state_dict['position_encoding.pe'].shape[1]
        
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
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=8,  # Default
            n_layers=n_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=0.1,  # Default
            activation_type='swiglu'
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model and tokenizer loaded from: {path}")
    
    return model, tokenizer