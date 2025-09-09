import sentencepiece as spm
import os
from pathlib import Path
from typing import List, Union
import tempfile


class SubwordTokenizer:
    """
    Subword tokenizer using SentencePiece for Slovenian poetry.
    
    Handles morphologically rich South Slavic languages better than character-level
    tokenization by learning subword units that capture morphological patterns.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize SubwordTokenizer.
        
        Args:
            model_path: Path to trained SentencePiece model file (.model)
        """
        self.sp = spm.SentencePieceProcessor()
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.sp.load(model_path)
            self.vocab_size = self.sp.vocab_size()
        else:
            self.vocab_size = None
    
    def train(self, text_data: Union[str, List[str]], vocab_size: int = 12000, 
              model_prefix: str = "poetry_tokenizer", save_dir: str = "models/tokenizer"):
        """
        Train SentencePiece tokenizer on poetry corpus.
        
        Args:
            text_data: Training text (string) or list of strings
            vocab_size: Target vocabulary size
            model_prefix: Prefix for saved model files
            save_dir: Directory to save model files
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Handle input data
        if isinstance(text_data, list):
            combined_text = '\n'.join(text_data)
        else:
            combined_text = text_data
        
        # Create temporary training file
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', 
                                       suffix='.txt', delete=False) as f:
            f.write(combined_text)
            train_file = f.name
        
        try:
            # Set model paths
            model_path = os.path.join(save_dir, model_prefix)
            self.model_path = f"{model_path}.model"
            
            # SentencePiece training arguments
            train_args = [
                f'--input={train_file}',
                f'--model_prefix={model_path}',
                f'--vocab_size={vocab_size}',
                '--model_type=bpe',  # Byte Pair Encoding
                '--character_coverage=0.9995',  # Good for European languages
                '--normalization_rule_name=nfkc',  # Unicode normalization
                '--remove_extra_whitespaces=true',
                '--split_by_unicode_script=true',
                '--split_by_whitespace=true',
                '--split_by_number=true',
                '--max_sentencepiece_length=16',
                '--shuffle_input_sentence=true',
                '--pad_id=0',
                '--unk_id=1', 
                '--bos_id=2',
                '--eos_id=3',
                '--user_defined_symbols=<PAD>,<UNK>,<BOS>,<EOS>',
            ]
            
            # Train the model
            spm.SentencePieceTrainer.train(' '.join(train_args))
            
            # Load the trained model
            self.sp.load(self.model_path)
            self.vocab_size = self.sp.vocab_size()
            
            print(f"Trained SentencePiece tokenizer:")
            print(f"   Model saved to: {self.model_path}")
            print(f"   Vocabulary size: {self.vocab_size:,}")
            print(f"   Training corpus: {len(combined_text):,} characters")
            
            # Show some example tokenizations
            sample_lines = combined_text.split('\n')[:5]
            print(f"\nSample tokenizations:")
            for line in sample_lines:
                if line.strip():
                    tokens = self.encode(line.strip(), add_special_tokens=False)
                    decoded = self.decode(tokens)
                    print(f"   '{line.strip()[:50]}...' → {tokens[:10]}... → '{decoded[:50]}...'")
                    break
        
        finally:
            # Clean up temporary file
            os.unlink(train_file)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to subword token IDs.
        
        Args:
            text: Input text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        if self.vocab_size is None:
            raise ValueError("Tokenizer not trained or loaded. Call train() or load model first.")
        
        token_ids = self.sp.encode(text)
        
        if add_special_tokens:
            bos_id = self.sp.bos_id()  # <BOS> = 2
            eos_id = self.sp.eos_id()  # <EOS> = 3
            token_ids = [bos_id] + token_ids + [eos_id]
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        if self.vocab_size is None:
            raise ValueError("Tokenizer not trained or loaded. Call train() or load model first.")
        
        return self.sp.decode(token_ids)
    
    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode multiple texts at once.
        
        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token ID lists
        """
        return [self.encode(text, add_special_tokens) for text in texts]
    
    def decode_batch(self, token_ids_list: List[List[int]]) -> List[str]:
        """
        Decode multiple token ID lists at once.
        
        Args:
            token_ids_list: List of token ID lists to decode
            
        Returns:
            List of decoded text strings
        """
        return [self.decode(token_ids) for token_ids in token_ids_list]
    
    def tokenize(self, text: str) -> List[str]:
        """
        Get actual subword tokens (strings) for inspection.
        
        Args:
            text: Input text
            
        Returns:
            List of subword token strings
        """
        if self.vocab_size is None:
            raise ValueError("Tokenizer not trained or loaded. Call train() or load model first.")
        
        return self.sp.encode(text, out_type=str)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def save(self, path: str):
        """Save tokenizer model to specified path."""
        if self.model_path and os.path.exists(self.model_path):
            import shutil
            shutil.copy(self.model_path, path)
            print(f"Tokenizer saved to: {path}")
        else:
            raise ValueError("No trained model to save")
    
    def load(self, path: str):
        """Load tokenizer model from specified path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer model not found: {path}")
        
        self.sp.load(path)
        self.vocab_size = self.sp.vocab_size()
        self.model_path = path
        print(f"Loaded tokenizer: {path} (vocab_size={self.vocab_size})")
    
    @property
    def pad_id(self) -> int:
        """Get padding token ID."""
        return self.sp.pad_id()  # 0
    
    @property
    def unk_id(self) -> int:
        """Get unknown token ID."""
        return self.sp.unk_id()  # 1
    
    @property  
    def bos_id(self) -> int:
        """Get beginning of sequence token ID."""
        return self.sp.bos_id()  # 2
    
    @property
    def eos_id(self) -> int:
        """Get end of sequence token ID."""
        return self.sp.eos_id()  # 3