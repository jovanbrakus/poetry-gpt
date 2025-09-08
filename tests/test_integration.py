import pytest
import torch
import tempfile
import os
from unittest.mock import patch

from poetry import train_model, generate_text
from model.transformer import PoetryGPT
from data.tokenizer import CharTokenizer
from training.config import TrainingConfig
from model.model_manager import save_model_and_tokenizer, load_model_and_tokenizer


class TestIntegration:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pt")
    
    def teardown_method(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_model_tokenizer_integration(self):
        text = "hello world this is a test"
        tokenizer = CharTokenizer(text)
        
        model = PoetryGPT(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=4,
            n_layers=2,
            max_len=16
        )
        
        encoded = tokenizer.encode(text[:10])
        input_ids = torch.tensor([encoded], dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_ids)
        
        assert output.shape == (1, len(encoded), tokenizer.vocab_size)
    
    def test_generation_integration(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        tokenizer = CharTokenizer(text)
        
        model = PoetryGPT(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=4,
            n_layers=2,
            max_len=50
        )
        
        prompt = "abc"
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=5)
        
        assert generated.shape[1] == len(prompt) + 5
        
        generated_text = tokenizer.decode(generated[0].tolist())
        assert generated_text.startswith(prompt)
    
    def test_save_load_integration(self):
        text = "test data for integration"
        tokenizer = CharTokenizer(text)
        config = TrainingConfig(d_model=32, n_heads=4, n_layers=2, max_len=16, d_ff=128)
        
        model = PoetryGPT(
            vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_len=config.max_len
        )
        
        save_model_and_tokenizer(model, tokenizer, config, self.model_path)
        
        loaded_model, loaded_tokenizer, loaded_config = load_model_and_tokenizer(self.model_path)
        
        test_input = torch.randint(0, tokenizer.vocab_size, (1, 5))
        
        with torch.no_grad():
            original_output = model(test_input)
            loaded_output = loaded_model(test_input)
        
        # Models should have same architecture and produce outputs of same shape
        assert original_output.shape == loaded_output.shape
        assert not torch.isnan(original_output).any()
        assert not torch.isnan(loaded_output).any()
    
    @patch('poetry.load_slovenian_data')
    @patch('poetry.train_gpt')
    @patch('poetry.save_model_and_tokenizer')
    def test_train_model_integration(self, mock_save, mock_train, mock_load_data):
        mock_text = "slovenian poetry text sample"
        mock_load_data.return_value = mock_text
        
        mock_model = PoetryGPT(vocab_size=20, d_model=32, n_heads=4, n_layers=2)
        mock_train.return_value = mock_model
        
        with patch('torch.backends.mps.is_available', return_value=False):
            train_model()
        
        mock_load_data.assert_called_once_with("static/slovenian")
        mock_train.assert_called_once()
        mock_save.assert_called_once()
    
    @patch('poetry.load_model_and_tokenizer')
    def test_generate_text_integration(self, mock_load):
        text = "test text for generation"
        tokenizer = CharTokenizer(text)
        config = TrainingConfig(d_model=32, n_heads=4, n_layers=2, d_ff=128)
        model = PoetryGPT(
            vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff
        )
        
        mock_load.return_value = (model, tokenizer, config)
        
        with patch('torch.backends.mps.is_available', return_value=False), \
             patch('builtins.print') as mock_print:
            
            generate_text("test")
        
        mock_load.assert_called_once()
    
    @patch('poetry.load_model_and_tokenizer')
    def test_generate_text_unknown_character(self, mock_load):
        text = "abc"
        tokenizer = CharTokenizer(text)
        config = TrainingConfig(d_model=32, n_heads=4, n_layers=2, d_ff=128)
        model = PoetryGPT(
            vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff
        )
        
        mock_load.return_value = (model, tokenizer, config)
        
        with patch('torch.backends.mps.is_available', return_value=False), \
             patch('builtins.print') as mock_print:
            
            generate_text("xyz")
        
        mock_print.assert_called()
        # Check if error message was printed (the function returns early on KeyError)
        printed_calls = [str(call) for call in mock_print.call_args_list]
        assert len(printed_calls) > 0
    
    def test_model_training_forward_pass(self):
        text = "sample training text with various characters"
        tokenizer = CharTokenizer(text)
        
        model = PoetryGPT(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            n_heads=8,
            n_layers=2,
            max_len=32
        )
        
        encoded_text = tokenizer.encode(text[:20])
        batch = torch.tensor([encoded_text], dtype=torch.long)
        
        x = batch[:, :-1]
        y = batch[:, 1:]
        
        logits = model(x)
        
        assert logits.shape == (1, x.shape[1], tokenizer.vocab_size)
        
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_end_to_end_tiny_training(self):
        text = "abcdefghijklmn" * 5
        tokenizer = CharTokenizer(text)
        config = TrainingConfig(
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            batch_size=1,
            seq_len=10,
            epochs=1,
            device=torch.device('cpu'),
            checkpoint_path=None  # Don't try to load existing checkpoint
        )
        
        model = PoetryGPT(
            vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_len=config.max_len
        )
        
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        from training.trainer import train_gpt
        trained_model = train_gpt(model, text, tokenizer, config)
        
        params_changed = False
        for name, param in trained_model.named_parameters():
            if not torch.equal(param, initial_params[name]):
                params_changed = True
                break
        
        assert params_changed, "Model parameters should change during training"
        
        prompt = "abc"
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        
        with torch.no_grad():
            generated = trained_model.generate(context, max_new_tokens=3)
        
        assert generated.shape[1] == len(prompt) + 3
        generated_text = tokenizer.decode(generated[0].tolist())
        assert generated_text.startswith(prompt)