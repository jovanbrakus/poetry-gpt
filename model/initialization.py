"""
Weight initialization utilities for better model training.
"""

import torch
import torch.nn as nn


def init_xavier_weights(module):
    """
    Initialize weights using Xavier/Glorot initialization.
    Good for layers with sigmoid, tanh, or no activation.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)


def init_kaiming_weights(module):
    """
    Initialize weights using Kaiming/He initialization.
    Good for layers with ReLU activation.
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')


def init_transformer_weights(module):
    """
    Initialize weights for transformer models.
    Uses standard GPT initialization: Normal(0, 0.02).
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def initialize_model(model, method='transformer'):
    """
    Initialize model weights.
    
    Args:
        model: PyTorch model to initialize
        method: 'xavier', 'kaiming', or 'transformer'
    
    Returns:
        The initialized model
    """
    if method == 'xavier':
        model.apply(init_xavier_weights)
    elif method == 'kaiming':
        model.apply(init_kaiming_weights)
    elif method == 'transformer':
        model.apply(init_transformer_weights)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'xavier', 'kaiming', or 'transformer'")
    
    return model