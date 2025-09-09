import torch
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Scaled dot-product attention with automatic Flash Attention optimization.
    
    Uses PyTorch's native scaled_dot_product_attention when available (PyTorch 2.0+),
    which automatically applies Flash Attention optimizations on supported hardware.
    Falls back to manual implementation for compatibility.
    
    Args:
        query: [batch_size * num_heads, seq_len, d_k]
        key: [batch_size * num_heads, seq_len, d_k] 
        value: [batch_size * num_heads, seq_len, d_k]
        mask: Optional causal mask [seq_len, seq_len]
        
    Returns:
        output: [batch_size * num_heads, seq_len, d_k]
        attention_weights: [batch_size * num_heads, seq_len, seq_len] (only for fallback)
    """
    # Try to use PyTorch's optimized SDPA (includes Flash Attention when available)
    if hasattr(F, 'scaled_dot_product_attention'):
        try:
            # Convert mask format for SDPA (expects True for positions to attend to)
            attn_mask = None
            if mask is not None:
                # Convert causal mask from [seq_len, seq_len] to proper format
                if mask.dim() == 2:
                    # Expand mask to match batch dimensions
                    batch_heads, seq_len, _ = query.shape
                    attn_mask = mask.bool().expand(batch_heads, seq_len, seq_len)
                else:
                    attn_mask = mask.bool()
            
            # Use PyTorch's optimized implementation
            output = F.scaled_dot_product_attention(
                query, key, value, 
                attn_mask=attn_mask,
                dropout_p=0.0,  # We handle dropout in the transformer block
                is_causal=attn_mask is None  # Auto-causal if no mask provided
            )
            
            # SDPA doesn't return attention weights, return None for compatibility
            return output, None
            
        except Exception as e:
            # Fall back to manual implementation if SDPA fails
            print(f"SDPA failed, falling back to manual attention: {e}")
            pass
    
    # Fallback to manual implementation
    d_k = query.size(-1)
    
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=query.dtype, device=query.device))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights