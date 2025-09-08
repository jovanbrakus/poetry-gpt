import pytest
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock

from training.config import TrainingConfig
from training.trainer import format_time, save_checkpoint, load_checkpoint, train_gpt
from model.transformer import MiniGPT
from data.tokenizer import CharTokenizer


class TestTrainingConfig:
    def test_default_values(self):
        config = TrainingConfig()
        assert config.d_model == 256
        assert config.n_heads == 8
        assert config.n_layers == 8
        assert config.d_ff == 1024
        assert config.max_len == 256
        assert config.dropout == 0.2
        assert config.batch_size == 32
        assert config.seq_len == 128
        assert config.epochs == 15
        assert config.learning_rate == 3e-4
        assert config.device == 'mps'
        assert config.checkpoint_path == "checkpoints/gpt_checkpoint.pt"
        assert config.save_every_epochs == 5
    
    def test_custom_values(self):
        config = TrainingConfig(
            d_model=512,
            n_heads=16,
            epochs=10,
            learning_rate=1e-3
        )
        assert config.d_model == 512
        assert config.n_heads == 16
        assert config.epochs == 10
        assert config.learning_rate == 1e-3


class TestTrainingUtils:
    def test_format_time_seconds_only(self):
        assert format_time(45) == "0m 45s"
    
    def test_format_time_minutes_and_seconds(self):
        assert format_time(125) == "2m 5s"
    
    def test_format_time_zero(self):
        assert format_time(0) == "0m 0s"
    
    def test_format_time_large_value(self):
        assert format_time(3661) == "61m 1s"


class TestCheckpointing:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint.pt")
    
    def teardown_method(self):
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
        os.rmdir(self.temp_dir)
    
    def test_save_checkpoint_creates_file(self):
        model = MiniGPT(vocab_size=50, d_model=64, n_heads=4, n_layers=2)
        optimizer = torch.optim.AdamW(model.parameters())
        
        save_checkpoint(model, optimizer, epoch=5, loss=0.5, path=self.checkpoint_path)
        
        assert os.path.exists(self.checkpoint_path)
    
    def test_save_checkpoint_content(self):
        model = MiniGPT(vocab_size=50, d_model=64, n_heads=4, n_layers=2)
        optimizer = torch.optim.AdamW(model.parameters())
        
        save_checkpoint(model, optimizer, epoch=10, loss=0.25, path=self.checkpoint_path)
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        assert checkpoint['epoch'] == 10
        assert checkpoint['loss'] == 0.25
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
    
    def test_load_checkpoint_nonexistent_file(self):
        model = MiniGPT(vocab_size=50, d_model=64, n_heads=4, n_layers=2)
        optimizer = torch.optim.AdamW(model.parameters())
        
        epoch = load_checkpoint(model, optimizer, path="nonexistent.pt")
        
        assert epoch == 0
    
    def test_load_checkpoint_existing_file(self):
        model = MiniGPT(vocab_size=50, d_model=64, n_heads=4, n_layers=2)
        optimizer = torch.optim.AdamW(model.parameters())
        
        save_checkpoint(model, optimizer, epoch=7, loss=0.3, path=self.checkpoint_path)
        
        new_model = MiniGPT(vocab_size=50, d_model=64, n_heads=4, n_layers=2)
        new_optimizer = torch.optim.AdamW(new_model.parameters())
        
        loaded_epoch = load_checkpoint(new_model, new_optimizer, path=self.checkpoint_path)
        
        assert loaded_epoch == 7


class TestTrainGPT:
    @pytest.fixture
    def sample_model(self):
        return MiniGPT(vocab_size=20, d_model=32, n_heads=4, n_layers=2, max_len=16)
    
    @pytest.fixture
    def sample_tokenizer(self):
        return CharTokenizer("abcdefghijklmnopqrst")
    
    @pytest.fixture
    def sample_config(self):
        config = TrainingConfig()
        config.device = torch.device('cpu')
        config.epochs = 2
        config.batch_size = 2
        config.seq_len = 8
        config.save_every_epochs = 1
        return config
    
    def test_train_gpt_basic_execution(self, sample_model, sample_tokenizer, sample_config):
        text_data = "abcdefghijklmnopqrst" * 10
        
        with patch('training.trainer.create_data_loader') as mock_loader, \
             patch('training.trainer.save_checkpoint') as mock_save, \
             patch('training.trainer.load_checkpoint', return_value=0) as mock_load:
            
            mock_batch = torch.randint(0, 20, (2, 9))
            mock_generator = MagicMock()
            mock_generator.__iter__ = MagicMock(return_value=iter([mock_batch, mock_batch]))
            mock_loader.return_value = lambda: mock_generator
            
            trained_model = train_gpt(sample_model, text_data, sample_tokenizer, sample_config)
            
            assert trained_model is not None
            mock_loader.assert_called_once()
            assert mock_save.call_count >= 1
    
    def test_train_gpt_model_parameters_updated(self, sample_model, sample_tokenizer, sample_config):
        initial_params = {name: param.clone() for name, param in sample_model.named_parameters()}
        text_data = "abcdefghijklmnopqrst" * 10
        
        with patch('training.trainer.create_data_loader') as mock_loader, \
             patch('training.trainer.save_checkpoint'), \
             patch('training.trainer.load_checkpoint', return_value=0):
            
            mock_batch = torch.randint(0, 20, (2, 9))
            mock_generator = MagicMock()
            mock_generator.__iter__ = MagicMock(return_value=iter([mock_batch, mock_batch]))
            mock_loader.return_value = lambda: mock_generator
            
            trained_model = train_gpt(sample_model, text_data, sample_tokenizer, sample_config)
            
            params_changed = False
            for name, param in trained_model.named_parameters():
                if not torch.equal(param, initial_params[name]):
                    params_changed = True
                    break
            
            assert params_changed, "Model parameters should have changed during training"
    
    def test_train_gpt_device_placement(self, sample_model, sample_tokenizer, sample_config):
        device = torch.device('cpu')
        sample_config.device = device
        text_data = "abcdefghijklmnopqrst" * 10
        
        with patch('training.trainer.create_data_loader') as mock_loader, \
             patch('training.trainer.save_checkpoint'), \
             patch('training.trainer.load_checkpoint', return_value=0):
            
            mock_batch = torch.randint(0, 20, (2, 9))
            mock_generator = MagicMock()
            mock_generator.__iter__ = MagicMock(return_value=iter([mock_batch]))
            mock_loader.return_value = lambda: mock_generator
            
            trained_model = train_gpt(sample_model, text_data, sample_tokenizer, sample_config)
            
            for param in trained_model.parameters():
                assert param.device == device