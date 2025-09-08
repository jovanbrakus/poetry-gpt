from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Model hyperparameters
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 1024
    max_len: int = 256
    dropout: float = 0.2
    
    # Training hyperparameters
    batch_size: int = 32
    seq_len: int = 128
    epochs: int = 15
    learning_rate: float = 3e-4
    
    # Device and checkpointing
    device: str = 'mps'  # 'mps' for Apple Silicon, 'cuda' for NVIDIA, 'cpu' for CPU
    checkpoint_path: str = "checkpoints/gpt_checkpoint.pt"
    save_every_epochs: int = 5