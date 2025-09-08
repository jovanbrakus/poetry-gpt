# Slovenian Poetry Generator 🇸🇮

A character-level GPT model trained on Slovenian poetry for generating poetic text in the Slovenian language.

## Model Architecture

**PoetryGPT** - A simplified GPT (Generative Pre-trained Transformer) implementation with:
- **Character-level tokenization** for handling Slovenian text and special characters
- **Multi-head attention** mechanism for capturing long-range dependencies
- **Transformer blocks** with residual connections and layer normalization
- **Causal masking** for autoregressive text generation

### Key Features
- **Modern Training**: Cosine learning rate scheduling with warmup, gradient clipping, weight decay
- **Flexible Architecture**: Configurable model size (layers, heads, dimensions)
- **Robust Generation**: Multiple sampling strategies (temperature, top-k)
- **Checkpointing**: Resume training from saved states

## Quick Start

### 1. Training a Model

Train the model on Slovenian poetry data:

```bash
python poetry.py train
```

The model will:
- Load poetry text files from `static/slovenian/`
- Save the complete model and tokenizer to `checkpoints/model_complete.pt`

### 2. Generating Poetry

Generate poetry from a prompt:

```bash
python poetry.py generate "V tihih"
```

Example output:
```
V tihih večernih urah,
ko sonce zahaja za hribe,
se spomnim na stare čase...
```

## Installation

### Requirements
- Python 3.11+
- PyTorch 2.8+
- Other dependencies in `pyproject.toml`

### Setup
```bash
# Install dependencies
uv sync
```


### Advanced Configuration

Customize training parameters by modifying `TrainingConfig`:

```python
from training import TrainingConfig

config = TrainingConfig(
    # Model architecture
    d_model=512,        # Model dimension
    n_heads=8,          # Number of attention heads
    n_layers=12,        # Number of transformer layers
    
    # Training parameters
    epochs=20,
    learning_rate=1e-3,
    batch_size=64,
    
    # Optimization features
    use_lr_scheduler=True,    # Cosine scheduling with warmup
    gradient_clip_val=1.0,    # Gradient clipping
    weight_decay=0.01,        # L2 regularization
)
```

## Model Details

### Architecture Specs
- **Default Config**: 256d model, 8 heads, 8 layers (~6.4M parameters)
- **Context Length**: 256 tokens
- **Vocabulary**: Character-level (typically ~160 unique characters)
- **Training Data**: Slovenian poetry corpus from `static/slovenian/`

### Generation Parameters
- **Temperature**: Controls randomness (0.1 = conservative, 2.0 = creative)
- **Top-k**: Limits sampling to k most likely tokens
- **Max tokens**: Maximum length of generated text

### Training Features
- **Learning Rate Scheduling**: Cosine annealing with linear warmup
- **Gradient Clipping**: Prevents exploding gradients
- **Checkpointing**: Automatic saving every 5 epochs
- **Progress Tracking**: Real-time loss and learning rate monitoring

## File Structure

```
poezija/
├── model/              # Model architecture
│   ├── transformer.py  # Main PoetryGPT implementation
│   ├── attention.py    # Attention mechanisms
│   └── ...
├── data/              # Data handling
│   ├── tokenizer.py   # Character tokenizer
│   ├── loader.py      # Data loading utilities
│   └── ...
├── training/          # Training infrastructure
│   ├── trainer.py     # Main training loop
│   ├── scheduler.py   # Learning rate scheduling
│   └── config.py      # Training configuration
├── static/slovenian/  # Poetry text files
├── tests/             # Unit tests
├── poetry.py          # Main CLI script
└── README.md
```

## Development

### Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test category
uv run pytest tests/test_model.py -v
```
