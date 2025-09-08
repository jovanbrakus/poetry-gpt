import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RoPEPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    
    RoPE applies rotary transformations to query and key embeddings based on their
    position, enabling the model to naturally understand relative positions.
    
    Used in models like LLaMA, GPT-NeoX, and PaLM.
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length (for precomputing frequencies)
        base: Base for frequency calculation (default: 10000)
        device: Device to place tensors on
    """
    
    def __init__(self, d_model: int, max_len: int = 2048, base: float = 10000.0, device=None):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # RoPE is typically applied to query and key, so we use d_head
        # For multi-head attention, d_head = d_model / num_heads
        # We'll compute for the full d_model and slice as needed
        
        # Compute frequency for each dimension pair
        # RoPE operates on pairs of dimensions
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        
        # Register as buffer so it moves with the model but isn't a parameter
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute rotary embeddings for efficiency
        self._precompute_freqs_cis(max_len, device)
    
    def _precompute_freqs_cis(self, max_len: int, device=None):
        """Precompute complex exponentials for RoPE."""
        # Create position indices
        t = torch.arange(max_len, device=device, dtype=self.inv_freq.dtype)
        
        # Compute frequencies for all positions and dimensions
        # freqs shape: [max_len, d_model//2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Convert to complex exponentials: e^(i*theta) = cos(theta) + i*sin(theta)
        # We'll store cos and sin separately for efficiency
        cos_freqs = torch.cos(freqs)  # [max_len, d_model//2]
        sin_freqs = torch.sin(freqs)  # [max_len, d_model//2]
        
        # Register as buffers
        self.register_buffer('cos_freqs', cos_freqs)
        self.register_buffer('sin_freqs', sin_freqs)
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate the second half of dimensions.
        
        For RoPE, we split dimensions into pairs and apply rotation.
        This function swaps and negates the second half for the rotation formula.
        
        Args:
            x: Input tensor [..., d_model]
            
        Returns:
            Rotated tensor with same shape
        """
        d = x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rope(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> torch.Tensor:
        """
        Apply RoPE to input tensor.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model] or [batch_size, num_heads, seq_len, d_head]
            seq_len: Sequence length
            offset: Position offset (for caching/generation)
            
        Returns:
            Tensor with RoPE applied
        """
        # Handle different input shapes
        original_shape = x.shape
        if len(original_shape) == 4:
            # Multi-head case: [batch_size, num_heads, seq_len, d_head]
            batch_size, num_heads, seq_len, d_head = original_shape
            x = x.reshape(batch_size * num_heads, seq_len, d_head)
        
        # Get precomputed cos/sin values for the sequence
        cos = self.cos_freqs[offset:offset + seq_len, :]  # [seq_len, d_model//2]
        sin = self.sin_freqs[offset:offset + seq_len, :]  # [seq_len, d_model//2]
        
        # Expand cos and sin to match input dimensions
        d_model = x.shape[-1]
        if cos.shape[-1] != d_model // 2:
            # If d_head != d_model (multi-head case), we need to slice
            cos = cos[..., :d_model//2]
            sin = sin[..., :d_model//2]
        
        # Repeat each frequency for the pair of dimensions
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # [seq_len, d_model]
        sin = torch.repeat_interleave(sin, 2, dim=-1)  # [seq_len, d_model]
        
        # Apply rotation: x * cos + rotate_half(x) * sin
        x_rotated = x * cos.unsqueeze(0) + self.rotate_half(x) * sin.unsqueeze(0)
        
        # Restore original shape if needed
        if len(original_shape) == 4:
            x_rotated = x_rotated.reshape(original_shape)
        
        return x_rotated
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                seq_len: Optional[int] = None, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.
        
        Args:
            query: Query tensor [..., seq_len, d_model]
            key: Key tensor [..., seq_len, d_model] 
            seq_len: Sequence length (inferred if None)
            offset: Position offset
            
        Returns:
            Tuple of (rotated_query, rotated_key)
        """
        if seq_len is None:
            seq_len = query.shape[-2]
        
        # Extend precomputed frequencies if needed
        if offset + seq_len > self.max_len:
            self._precompute_freqs_cis(offset + seq_len, device=query.device)
        
        # Apply RoPE to query and key
        query_rotated = self.apply_rope(query, seq_len, offset)
        key_rotated = self.apply_rope(key, seq_len, offset)
        
        return query_rotated, key_rotated


def create_rope_cache(d_model: int, max_len: int = 2048, base: float = 10000.0, device=None):
    """
    Utility function to create RoPE frequency cache.
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        base: Base for frequency calculation
        device: Device to place tensors on
        
    Returns:
        Tuple of (cos_freqs, sin_freqs) tensors
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
    if device is not None:
        inv_freq = inv_freq.to(device)
    
    t = torch.arange(max_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)
    
    return cos_freqs, sin_freqs


def apply_rope_simple(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Simple RoPE application for testing.
    
    Args:
        x: Input tensor [..., seq_len, d_model]
        cos: Cosine frequencies [seq_len, d_model//2] or [seq_len, d_model]
        sin: Sine frequencies [seq_len, d_model//2] or [seq_len, d_model]
        
    Returns:
        Rotated tensor
    """
    def rotate_half(x):
        d = x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    # Ensure cos and sin have the right shape
    if cos.shape[-1] == x.shape[-1] // 2:
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
    
    return x * cos + rotate_half(x) * sin