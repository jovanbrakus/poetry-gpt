import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .tokenizer import SubwordTokenizer


def load_slovenian_data(data_path="static/slovenian"):
    """Load and concatenate all Slovenian poetry text files"""
    text_data = []
    
    slovenian_path = Path(data_path)
    
    if not slovenian_path.exists():
        raise FileNotFoundError(f"Slovenian data directory not found: {data_path}")
    
    txt_files = list(slovenian_path.glob("*.txt"))
    
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_path}")
    
    for txt_file in sorted(txt_files):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                text_data.append(content)
                print(f"Loaded: {txt_file.name} ({len(content):,} characters)")
        except Exception as e:
            print(f"Error loading {txt_file.name}: {e}")
            continue
    
    combined_text = '\n\n'.join(text_data)
    print(f"\nTotal loaded: {len(txt_files)} files, {len(combined_text):,} characters")
    
    return combined_text


def load_serbian_data(data_path="static/serbian"):
    """Load and concatenate all Serbian poetry text files"""
    text_data = []
    
    serbian_path = Path(data_path)
    
    if not serbian_path.exists():
        raise FileNotFoundError(f"Serbian data directory not found: {data_path}")
    
    txt_files = list(serbian_path.glob("*.txt"))
    
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_path}")
    
    for txt_file in sorted(txt_files):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                text_data.append(content)
                print(f"Loaded: {txt_file.name} ({len(content):,} characters)")
        except Exception as e:
            print(f"Error loading {txt_file.name}: {e}")
            continue
    
    combined_text = '\n\n'.join(text_data)
    print(f"\nTotal loaded: {len(txt_files)} files, {len(combined_text):,} characters")
    
    return combined_text


def load_combined_data(slovenian_path="static/slovenian", serbian_path=None):
    """Load Slovenian poetry data (Serbian disabled by default)"""
    print("Loading Slovenian poetry data...")
    slovenian_text = load_slovenian_data(slovenian_path)
    
    print(f"\nPoetry corpus:")
    print(f"   Slovenian: {len(slovenian_text):,} characters")
    print(f"   Total: {len(slovenian_text):,} characters")
    
    return slovenian_text


def prepare_tokenizer(text_data: str, vocab_size: int = 12000, 
                     model_name: str = "poetry_tokenizer",
                     save_dir: str = "static/models/tokenizer") -> SubwordTokenizer:
    """
    Train or load subword tokenizer for poetry data.
    
    Args:
        text_data: Combined training text
        vocab_size: Target vocabulary size for tokenizer
        model_name: Name for the tokenizer model
        save_dir: Directory to save/load tokenizer
        
    Returns:
        Trained SubwordTokenizer instance
    """
    model_path = os.path.join(save_dir, f"{model_name}.model")
    
    tokenizer = SubwordTokenizer()
    
    # Check if tokenizer already exists
    if os.path.exists(model_path):
        print(f"Loading existing tokenizer: {model_path}")
        tokenizer.load(model_path)
    else:
        print(f"Training new subword tokenizer...")
        tokenizer.train(
            text_data=text_data,
            vocab_size=vocab_size,
            model_prefix=model_name,
            save_dir=save_dir
        )
    
    return tokenizer


class PoetryDataset(Dataset):
    """
    PyTorch Dataset for poetry text using subword tokenization.
    
    Creates overlapping sequences of subword tokens for language modeling.
    """
    
    def __init__(self, text_data: str, tokenizer: SubwordTokenizer, 
                 seq_length: int = 256, stride: int = None):
        """
        Initialize PoetryDataset.
        
        Args:
            text_data: Raw text data to tokenize
            tokenizer: Trained SubwordTokenizer instance
            seq_length: Maximum sequence length for model input
            stride: Step size for creating overlapping sequences (defaults to seq_length)
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride or seq_length
        
        print(f"Tokenizing poetry corpus...")
        
        # Tokenize the entire corpus
        self.token_ids = tokenizer.encode(text_data, add_special_tokens=False)
        
        print(f"   Original text: {len(text_data):,} characters")
        print(f"   Tokenized: {len(self.token_ids):,} tokens")
        print(f"   Compression ratio: {len(text_data) / len(self.token_ids):.1f}x")
        
        # Create overlapping sequences
        self.sequences = []
        for i in range(0, len(self.token_ids) - seq_length, self.stride):
            sequence = self.token_ids[i:i + seq_length + 1]  # +1 for next token prediction
            self.sequences.append(sequence)
        
        print(f"   Created {len(self.sequences):,} training sequences")
        print(f"   Sequence length: {seq_length} tokens")
        print(f"   Stride: {self.stride} tokens")
        
        # Show tokenization examples
        self._show_tokenization_examples(text_data)
    
    def _show_tokenization_examples(self, text_data: str, num_examples: int = 3):
        """Show some tokenization examples for debugging"""
        print(f"\nTokenization examples:")
        
        lines = [line.strip() for line in text_data.split('\n') if line.strip()][:num_examples]
        
        for i, line in enumerate(lines, 1):
            # Show first 80 characters
            display_text = line[:80] + "..." if len(line) > 80 else line
            
            # Tokenize and show tokens
            tokens = self.tokenizer.tokenize(line)
            token_ids = self.tokenizer.encode(line, add_special_tokens=False)
            
            print(f"   Example {i}:")
            print(f"     Text: '{display_text}'")
            print(f"     Tokens: {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
            print(f"     IDs: {token_ids[:15]}{'...' if len(token_ids) > 15 else ''}")
            print(f"     Length: {len(token_ids)} tokens (vs {len(line)} chars)")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a training example.
        
        Returns:
            dict with 'input_ids' and 'labels' tensors
        """
        sequence = self.sequences[idx]
        
        # Input is all tokens except the last
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        
        # Labels are all tokens except the first (shifted by 1 for next token prediction)
        labels = torch.tensor(sequence[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }
    
    def get_vocab_size(self):
        """Get tokenizer vocabulary size"""
        return self.tokenizer.get_vocab_size()
    
    def decode_sequence(self, token_ids):
        """Decode a sequence of token IDs back to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids)


def create_poetry_dataset(slovenian_path="static/slovenian", 
                         vocab_size=12000,
                         seq_length=256,
                         stride=None,
                         tokenizer_name="poetry_tokenizer") -> PoetryDataset:
    """
    Convenience function to create a complete poetry dataset.
    
    Args:
        slovenian_path: Path to Slovenian poetry files
        vocab_size: Subword vocabulary size
        seq_length: Maximum sequence length
        stride: Stride for overlapping sequences
        tokenizer_name: Name for tokenizer model files
        
    Returns:
        PoetryDataset instance ready for training
    """
    print("Creating Poetry Dataset with Subword Tokenization")
    print("=" * 60)
    
    # Load Slovenian text data only
    text_data = load_combined_data(slovenian_path)
    
    # Prepare tokenizer
    tokenizer = prepare_tokenizer(text_data, vocab_size, tokenizer_name)
    
    # Create dataset
    dataset = PoetryDataset(text_data, tokenizer, seq_length, stride)
    
    print("=" * 60)
    print(f"Dataset ready! Vocabulary: {tokenizer.get_vocab_size():,} subwords")
    
    return dataset