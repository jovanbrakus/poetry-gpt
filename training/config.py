from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Model hyperparameters - Scaled for larger dataset (9.76M chars)
    d_model: int = 384  # Increased from 256
    n_heads: int = 8
    n_layers: int = 12  # Increased from 8
    d_ff: int = 1536    # Increased from 1024
    max_len: int = 384  # Increased from 256 for more context
    dropout: float = 0.2
    
    # Tokenization - Enhanced for larger dataset
    vocab_size: int = 16000  # Increased from 12000
    tokenizer_name: str = "poetry_tokenizer"
    
    # Training hyperparameters - Adjusted for larger model
    batch_size: int = 64    # Increased from 32 (reduce if memory issues)
    seq_len: int = 384      # Increased from 256
    epochs: int = 25        # Increased from 15
    learning_rate: float = 1e-4  # Reduced from 3e-4 for stability
    
    # Learning rate scheduling - Adjusted for longer training
    use_lr_scheduler: bool = True
    warmup_steps: int = 2000  # Increased warmup for larger model
    min_lr_ratio: float = 0.1  # Minimum LR as ratio of max LR
    
    # Optimization
    gradient_clip_val: float = 1.0
    weight_decay: float = 0.01
    
    # Model initialization and architecture
    init_method: str = 'transformer'  # 'xavier', 'kaiming', or 'transformer'
    activation_type: str = 'swiglu'   # 'swiglu', 'geglu', 'reglu', 'relu'
    
    # Validation and early stopping
    validation_split: float = 0.1  # 10% of data for validation
    early_stopping_patience: int = 5  # Stop if no improvement for 5 epochs
    min_delta: float = 0.001  # Minimum change to qualify as improvement
    
    # Device and checkpointing
    device: str = 'mps'
    checkpoint_path: str = "static/checkpoints/gpt_checkpoint.pt"
    save_every_epochs: int = 3  # Save more frequently for longer training