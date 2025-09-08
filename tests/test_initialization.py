import pytest
import torch
import torch.nn as nn
from model.initialization import (
    init_xavier_weights, 
    init_kaiming_weights, 
    init_transformer_weights, 
    initialize_model
)
from model.transformer import PoetryGPT


class TestWeightInitialization:
    
    def test_init_xavier_weights_linear(self):
        """Test Xavier initialization for linear layers"""
        layer = nn.Linear(10, 5)
        
        # Initialize with Xavier
        init_xavier_weights(layer)
        
        # Check that weights were initialized (not default)
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        
        # Check bias is zero
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))
        
        # Check Xavier bounds approximately
        fan_in, fan_out = 10, 5
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        assert torch.all(torch.abs(layer.weight) <= bound * 1.1)  # Small tolerance
    
    def test_init_xavier_weights_embedding(self):
        """Test Xavier initialization for embedding layers"""
        layer = nn.Embedding(100, 50)
        
        # Initialize with Xavier
        init_xavier_weights(layer)
        
        # Check that weights were initialized
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
    
    def test_init_kaiming_weights_linear(self):
        """Test Kaiming initialization for linear layers"""
        layer = nn.Linear(10, 5)
        
        # Initialize with Kaiming
        init_kaiming_weights(layer)
        
        # Check that weights were initialized
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        
        # Check bias is zero
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))
    
    def test_init_kaiming_weights_embedding(self):
        """Test Kaiming initialization for embedding layers"""
        layer = nn.Embedding(100, 50)
        
        # Initialize with Kaiming
        init_kaiming_weights(layer)
        
        # Check that weights were initialized
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
    
    def test_init_transformer_weights_linear(self):
        """Test transformer initialization for linear layers"""
        layer = nn.Linear(10, 5)
        
        # Initialize with transformer method
        init_transformer_weights(layer)
        
        # Check that weights were initialized
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        
        # Check bias is zero
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))
        
        # Check standard deviation is approximately 0.02
        std = torch.std(layer.weight)
        assert 0.01 < std < 0.05  # Reasonable range around 0.02
    
    def test_init_transformer_weights_embedding(self):
        """Test transformer initialization for embedding layers"""
        layer = nn.Embedding(100, 50)
        
        # Initialize with transformer method
        init_transformer_weights(layer)
        
        # Check that weights were initialized
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        
        # Check standard deviation
        std = torch.std(layer.weight)
        assert 0.01 < std < 0.05
    
    def test_init_transformer_weights_layernorm(self):
        """Test transformer initialization for LayerNorm"""
        layer = nn.LayerNorm(50)
        
        # Initialize with transformer method
        init_transformer_weights(layer)
        
        # Check LayerNorm weight is ones
        assert torch.allclose(layer.weight, torch.ones_like(layer.weight))
        
        # Check LayerNorm bias is zeros
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))
    
    def test_initialize_model_xavier(self):
        """Test full model initialization with Xavier"""
        model = PoetryGPT(vocab_size=50, d_model=32, n_heads=4, n_layers=2, init_method='xavier')
        
        # Check that some weights are not zero (initialized)
        total_params = sum(p.numel() for p in model.parameters())
        zero_params = sum((p == 0).sum().item() for p in model.parameters())
        
        # Should have many non-zero parameters
        assert zero_params < total_params * 0.5
    
    def test_initialize_model_kaiming(self):
        """Test full model initialization with Kaiming"""
        model = PoetryGPT(vocab_size=50, d_model=32, n_heads=4, n_layers=2, init_method='kaiming')
        
        # Check that some weights are not zero
        total_params = sum(p.numel() for p in model.parameters())
        zero_params = sum((p == 0).sum().item() for p in model.parameters())
        
        assert zero_params < total_params * 0.5
    
    def test_initialize_model_transformer(self):
        """Test full model initialization with transformer method"""
        model = PoetryGPT(vocab_size=50, d_model=32, n_heads=4, n_layers=2, init_method='transformer')
        
        # Check that some weights are not zero
        total_params = sum(p.numel() for p in model.parameters())
        zero_params = sum((p == 0).sum().item() for p in model.parameters())
        
        assert zero_params < total_params * 0.5
    
    def test_initialize_model_invalid_method(self):
        """Test that invalid initialization method raises error"""
        with pytest.raises(ValueError, match="Unknown method"):
            initialize_model(nn.Linear(10, 5), method='invalid_method')
    
    def test_different_initialization_methods_differ(self):
        """Test that different initialization methods produce different results"""
        # Create identical models
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)
        model3 = nn.Linear(10, 5)
        
        # Initialize with different methods
        init_xavier_weights(model1)
        init_kaiming_weights(model2)
        init_transformer_weights(model3)
        
        # Check that they produce different weights
        assert not torch.allclose(model1.weight, model2.weight)
        assert not torch.allclose(model1.weight, model3.weight)
        assert not torch.allclose(model2.weight, model3.weight)
    
    def test_model_with_initialization_parameter(self):
        """Test that model initialization parameter works"""
        # Test each method
        for method in ['xavier', 'kaiming', 'transformer']:
            model = PoetryGPT(
                vocab_size=50, 
                d_model=32, 
                n_heads=4, 
                n_layers=2,
                init_method=method
            )
            
            # Check that model was created without errors
            assert model is not None
            
            # Check that weights are not all zeros
            has_nonzero_weights = any(
                not torch.allclose(p, torch.zeros_like(p)) 
                for p in model.parameters()
            )
            assert has_nonzero_weights