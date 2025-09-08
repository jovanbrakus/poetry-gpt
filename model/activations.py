import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x: torch.Tensor) -> torch.Tensor:
    """
    Swish activation function (also known as SiLU - Sigmoid Linear Unit).
    
    Formula: x * sigmoid(x)
    
    Args:
        x: Input tensor
        
    Returns:
        Swish-activated tensor
    """
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU).
    
    A gated activation function that uses Swish activation.
    Used in modern models like PaLM, LLaMA, etc.
    
    Formula: swish(gate_proj(x)) * value_proj(x)
    
    Args:
        d_model (int): Input dimension
        d_ff (int): Feed-forward dimension (will be split for gate and value)
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        
        # For SwiGLU, we typically use 2/3 * d_ff to maintain similar parameter count
        # as the original ReLU FFN, but here we'll use the full d_ff as specified
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.value_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Output tensor of shape (..., d_model)
        """
        gate = swish(self.gate_proj(x))
        value = self.value_proj(x)
        intermediate = gate * value
        intermediate = self.dropout(intermediate)
        return self.down_proj(intermediate)


class GeGLU(nn.Module):
    """
    GELU-Gated Linear Unit (GeGLU).
    
    Alternative to SwiGLU using GELU instead of Swish.
    
    Formula: gelu(gate_proj(x)) * value_proj(x)
    
    Args:
        d_model (int): Input dimension
        d_ff (int): Feed-forward dimension
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.value_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GeGLU activation.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Output tensor of shape (..., d_model)
        """
        gate = F.gelu(self.gate_proj(x))
        value = self.value_proj(x)
        intermediate = gate * value
        intermediate = self.dropout(intermediate)
        return self.down_proj(intermediate)


class ReGLU(nn.Module):
    """
    ReLU-Gated Linear Unit (ReGLU).
    
    Traditional gated unit using ReLU (for comparison/fallback).
    
    Formula: relu(gate_proj(x)) * value_proj(x)
    
    Args:
        d_model (int): Input dimension
        d_ff (int): Feed-forward dimension
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.value_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ReGLU activation.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Output tensor of shape (..., d_model)
        """
        gate = F.relu(self.gate_proj(x))
        value = self.value_proj(x)
        intermediate = gate * value
        intermediate = self.dropout(intermediate)
        return self.down_proj(intermediate)


def create_feedforward(d_model: int, d_ff: int, dropout: float = 0.0, 
                      activation_type: str = 'swiglu') -> nn.Module:
    """
    Create a feed-forward layer with specified activation type.
    
    Args:
        d_model (int): Input/output dimension
        d_ff (int): Feed-forward hidden dimension
        dropout (float): Dropout probability
        activation_type (str): Type of activation ('swiglu', 'geglu', 'reglu', 'relu')
        
    Returns:
        Feed-forward module
        
    Raises:
        ValueError: If activation_type is not supported
    """
    activation_type = activation_type.lower()
    
    if activation_type == 'swiglu':
        return SwiGLU(d_model, d_ff, dropout)
    elif activation_type == 'geglu':
        return GeGLU(d_model, d_ff, dropout)
    elif activation_type == 'reglu':
        return ReGLU(d_model, d_ff, dropout)
    elif activation_type == 'relu':
        # Traditional ReLU FFN for backward compatibility
        return nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}. "
                        f"Supported types: 'swiglu', 'geglu', 'reglu', 'relu'")