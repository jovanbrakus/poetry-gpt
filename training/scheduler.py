import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with linear warmup.
    
    The learning rate starts at 0, linearly increases to base_lr during warmup,
    then follows cosine annealing decay from base_lr to min_lr.
    """
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)  # Clamp to [0, 1]
            
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            lr_range = 1.0 - self.min_lr_ratio
            lr_multiplier = self.min_lr_ratio + lr_range * cosine_factor
            
            return [base_lr * lr_multiplier for base_lr in self.base_lrs]


def get_lr_scheduler(optimizer, config, total_steps):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: TrainingConfig object
        total_steps: Total number of training steps
    
    Returns:
        Learning rate scheduler or None if not enabled
    """
    if not config.use_lr_scheduler:
        return None
    
    return CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=config.min_lr_ratio
    )


def clip_gradients(model, max_norm):
    """
    Clip gradients to prevent exploding gradients.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
    
    Returns:
        Total norm of gradients before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)