import torch


def create_causal_mask(seq_len, device):
    """Create a mask to prevent attention to future positions"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.to(device)
    return mask == 0