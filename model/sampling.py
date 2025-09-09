"""
Improved sampling methods for text generation.
"""

import torch
import torch.nn.functional as F


def top_k_sampling(logits, k):
    """
    Apply top-k sampling to logits.
    
    Args:
        logits: Tensor of shape [batch_size, vocab_size]
        k: Number of top tokens to keep
    
    Returns:
        Filtered logits with only top-k tokens
    """
    if k <= 0:
        return logits
    
    # Get top-k values and indices
    top_k_values, top_k_indices = torch.topk(logits, min(k, logits.size(-1)))
    
    # Create mask for top-k tokens
    mask = torch.full_like(logits, -float('inf'))
    mask.scatter_(-1, top_k_indices, top_k_values)
    
    return mask


def top_p_sampling(logits, p):
    """
    Apply top-p (nucleus) sampling to logits.
    
    Args:
        logits: Tensor of shape [batch_size, vocab_size]
        p: Cumulative probability threshold (0.0 to 1.0)
    
    Returns:
        Filtered logits with nucleus sampling applied
    """
    if p <= 0.0 or p >= 1.0:
        return logits
    
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Create mask for tokens outside nucleus
    sorted_indices_to_remove = cumulative_probs > p
    
    # Keep at least one token (the first one)
    sorted_indices_to_remove[..., 0] = False
    
    # Create mask in original order
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    
    # Apply mask
    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = -float('inf')
    
    return filtered_logits


def apply_repetition_penalty(logits, input_ids, penalty=1.0):
    """
    Apply repetition penalty to logits based on previously generated tokens.
    
    Args:
        logits: Tensor of shape [batch_size, vocab_size]
        input_ids: Tensor of shape [batch_size, seq_len] with previous tokens
        penalty: Repetition penalty factor (>1.0 discourages repetition)
    
    Returns:
        Logits with repetition penalty applied
    """
    if penalty == 1.0:
        return logits
    
    batch_size, vocab_size = logits.shape
    
    # Apply penalty to tokens that appear in input_ids
    for batch_idx in range(batch_size):
        for token_id in input_ids[batch_idx]:
            token_id = token_id.item()
            if 0 <= token_id < vocab_size:
                # If logit is positive, divide by penalty (reduce probability)
                # If logit is negative, multiply by penalty (reduce probability further)
                if logits[batch_idx, token_id] > 0:
                    logits[batch_idx, token_id] /= penalty
                else:
                    logits[batch_idx, token_id] *= penalty
    
    return logits


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, 
                      input_ids=None, repetition_penalty=1.0, vocab_size=None, 
                      exclude_tokens=None):
    """
    Sample tokens from logits with various sampling strategies.
    
    Args:
        logits: Tensor of shape [batch_size, vocab_size]
        temperature: Temperature for sampling (higher = more random)
        top_k: Top-k filtering (None to disable)
        top_p: Top-p/nucleus sampling (None to disable) 
        input_ids: Previous tokens for repetition penalty (None to disable)
        repetition_penalty: Repetition penalty factor (1.0 to disable)
        vocab_size: Maximum valid token ID (for safety clamping)
        exclude_tokens: List of token IDs to exclude from sampling (e.g., UNK token)
    
    Returns:
        Sampled token indices of shape [batch_size, 1]
    """
    # Exclude specific tokens (like UNK) by setting their logits to -inf
    if exclude_tokens is not None:
        for token_id in exclude_tokens:
            if 0 <= token_id < logits.size(-1):
                logits[:, token_id] = -float('inf')
    
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Apply repetition penalty
    if input_ids is not None and repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
    
    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        logits = top_k_sampling(logits, top_k)
    
    # Apply top-p filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        logits = top_p_sampling(logits, top_p)
    
    # Convert to probabilities and sample
    probs = F.softmax(logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
    
    # Safety check: clamp token IDs to valid vocabulary range
    if vocab_size is not None:
        next_tokens = torch.clamp(next_tokens, 0, vocab_size - 1)
    
    return next_tokens