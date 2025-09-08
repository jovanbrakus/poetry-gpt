import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    More efficient than LayerNorm as it doesn't compute mean and doesn't have bias.
    Used in modern models like LLaMA, PaLM, etc.
    
    Formula: x * rsqrt(mean(x²) + ε) * weight
    
    Args:
        d_model (int): The dimension of the input features
        eps (float): A value added to the denominator for numerical stability
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Normalized tensor of the same shape as input
        """
        # Compute RMS (Root Mean Square)
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        return x / rms * self.weight
    
    def extra_repr(self) -> str:
        return f"d_model={self.weight.shape[0]}, eps={self.eps}"