import pytest
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock

from model.model_manager import save_model_and_tokenizer, load_model_and_tokenizer
from model.transformer import MiniGPT
from data.tokenizer import CharTokenizer
from training.config import TrainingConfig


class TestModelManager:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = os.path.join(self.temp_dir, "test_model.pt")
    
    def teardown_method(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @pytest.fixture
    def sample_model(self):
        return MiniGPT(
            vocab_size=26,  # Match tokenizer vocab size
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=2048,
            max_len=128
        )
    
    @pytest.fixture
    def sample_tokenizer(self):
        return CharTokenizer("abcdefghijklmnopqrstuvwxyz")
    
    @pytest.fixture
    def sample_config(self):
        config = TrainingConfig()
        config.d_model = 64
        config.n_heads = 4
        config.n_layers = 2
        config.d_ff = 2048  # Match the default MiniGPT d_ff
        config.max_len = 128
        return config
    
    def test_save_model_and_tokenizer_creates_file(self, sample_model, sample_tokenizer, sample_config):
        save_model_and_tokenizer(sample_model, sample_tokenizer, sample_config, self.temp_path)
        
        assert os.path.exists(self.temp_path)
    
    def test_save_model_and_tokenizer_creates_directory(self, sample_model, sample_tokenizer, sample_config):
        nested_path = os.path.join(self.temp_dir, "nested", "deep", "model.pt")
        
        save_model_and_tokenizer(sample_model, sample_tokenizer, sample_config, nested_path)
        
        assert os.path.exists(nested_path)
    
    def test_save_model_and_tokenizer_content(self, sample_model, sample_tokenizer, sample_config):
        save_model_and_tokenizer(sample_model, sample_tokenizer, sample_config, self.temp_path)
        
        checkpoint = torch.load(self.temp_path, map_location='cpu', weights_only=False)
        
        assert 'model_state_dict' in checkpoint
        assert 'tokenizer' in checkpoint
        assert 'config' in checkpoint
        assert 'vocab_size' in checkpoint
        assert checkpoint['vocab_size'] == sample_tokenizer.vocab_size
    
    def test_load_model_and_tokenizer_nonexistent_file(self):
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_model_and_tokenizer("/nonexistent/path.pt")
    
    def test_load_model_and_tokenizer_success(self, sample_model, sample_tokenizer, sample_config):
        save_model_and_tokenizer(sample_model, sample_tokenizer, sample_config, self.temp_path)
        
        loaded_model, loaded_tokenizer, loaded_config = load_model_and_tokenizer(self.temp_path)
        
        assert isinstance(loaded_model, MiniGPT)
        assert isinstance(loaded_tokenizer, CharTokenizer)
        assert isinstance(loaded_config, TrainingConfig)
    
    def test_load_model_and_tokenizer_model_architecture(self, sample_model, sample_tokenizer, sample_config):
        save_model_and_tokenizer(sample_model, sample_tokenizer, sample_config, self.temp_path)
        
        loaded_model, loaded_tokenizer, loaded_config = load_model_and_tokenizer(self.temp_path)
        
        assert loaded_model.token_embedding.num_embeddings == sample_tokenizer.vocab_size
        assert loaded_model.token_embedding.embedding_dim == sample_config.d_model
        assert len(loaded_model.blocks) == sample_config.n_layers
        assert loaded_model.max_len == sample_config.max_len
    
    def test_load_model_and_tokenizer_tokenizer_preservation(self, sample_model, sample_tokenizer, sample_config):
        save_model_and_tokenizer(sample_model, sample_tokenizer, sample_config, self.temp_path)
        
        loaded_model, loaded_tokenizer, loaded_config = load_model_and_tokenizer(self.temp_path)
        
        assert loaded_tokenizer.vocab_size == sample_tokenizer.vocab_size
        assert loaded_tokenizer.char_to_idx == sample_tokenizer.char_to_idx
        assert loaded_tokenizer.idx_to_char == sample_tokenizer.idx_to_char
    
    def test_load_model_and_tokenizer_config_preservation(self, sample_model, sample_tokenizer, sample_config):
        save_model_and_tokenizer(sample_model, sample_tokenizer, sample_config, self.temp_path)
        
        loaded_model, loaded_tokenizer, loaded_config = load_model_and_tokenizer(self.temp_path)
        
        assert loaded_config.d_model == sample_config.d_model
        assert loaded_config.n_heads == sample_config.n_heads
        assert loaded_config.n_layers == sample_config.n_layers
        assert loaded_config.max_len == sample_config.max_len
    
    def test_save_load_roundtrip_model_state(self, sample_model, sample_tokenizer, sample_config):
        # Need to ensure the original model has the same architecture as what will be loaded
        original_model = MiniGPT(
            vocab_size=sample_tokenizer.vocab_size,
            d_model=sample_config.d_model,
            n_heads=sample_config.n_heads,
            n_layers=sample_config.n_layers,
            d_ff=sample_config.d_ff,
            max_len=sample_config.max_len
        )
        
        # Copy parameters from sample_model to original_model
        with torch.no_grad():
            original_model.token_embedding.weight.copy_(sample_model.token_embedding.weight[:sample_tokenizer.vocab_size])
            original_model.lm_head.weight.copy_(sample_model.lm_head.weight[:sample_tokenizer.vocab_size])
            original_model.lm_head.bias.copy_(sample_model.lm_head.bias[:sample_tokenizer.vocab_size])
        
        original_params = {name: param.clone() for name, param in original_model.named_parameters()}
        
        save_model_and_tokenizer(original_model, sample_tokenizer, sample_config, self.temp_path)
        loaded_model, _, _ = load_model_and_tokenizer(self.temp_path)
        
        for name, param in loaded_model.named_parameters():
            assert torch.equal(param, original_params[name]), f"Parameter {name} was not preserved"
    
    def test_save_load_tokenizer_functionality(self, sample_model, sample_tokenizer, sample_config):
        test_text = "hello"
        original_encoded = sample_tokenizer.encode(test_text)
        
        save_model_and_tokenizer(sample_model, sample_tokenizer, sample_config, self.temp_path)
        _, loaded_tokenizer, _ = load_model_and_tokenizer(self.temp_path)
        
        loaded_encoded = loaded_tokenizer.encode(test_text)
        assert loaded_encoded == original_encoded
        
        decoded = loaded_tokenizer.decode(loaded_encoded)
        assert decoded == test_text
    
    def test_default_path_parameter(self, sample_model, sample_tokenizer, sample_config):
        with patch('os.makedirs'), \
             patch('torch.save') as mock_save, \
             patch('builtins.print'):
            save_model_and_tokenizer(sample_model, sample_tokenizer, sample_config)
            mock_save.assert_called_once()
            args, kwargs = mock_save.call_args
            assert args[1] == "checkpoints/model_complete.pt"
    
    def test_load_default_path_parameter(self):
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="checkpoints/model_complete.pt"):
                load_model_and_tokenizer()