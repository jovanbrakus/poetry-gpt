from dataclasses import dataclass


@dataclass
class TrainingConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 1024
    max_len: int = 384
    dropout: float = 0.2
    
    vocab_size: int = 8000
    tokenizer_name: str = "poetry_tokenizer"
    
    batch_size: int = 32
    seq_len: int = 384
    epochs: int = 25
    learning_rate: float = 1e-4
    
    use_lr_scheduler: bool = True
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1
    
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
    save_every_epochs: int = 5