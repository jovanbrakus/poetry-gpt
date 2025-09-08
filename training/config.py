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
    
    # Tokenization
    vocab_size: int = 12000  # Subword vocabulary size
    tokenizer_name: str = "poetry_tokenizer"
    
    # Training hyperparameters
    batch_size: int = 32
    seq_len: int = 256  # Longer sequences possible with subwords
    epochs: int = 15
    learning_rate: float = 3e-4
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1  # Minimum LR as ratio of max LR
    
    # Optimization
    gradient_clip_val: float = 1.0
    weight_decay: float = 0.01
    
    # Model initialization and architecture
    init_method: str = 'transformer'  # 'xavier', 'kaiming', or 'transformer'
    activation_type: str = 'swiglu'   # 'swiglu', 'geglu', 'reglu', 'relu'
    
    # Device and checkpointing
    device: str = 'mps'
    checkpoint_path: str = "static/checkpoints/gpt_checkpoint.pt"
    save_every_epochs: int = 5