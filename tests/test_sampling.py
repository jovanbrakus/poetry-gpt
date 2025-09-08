import pytest
import torch
from model.sampling import (
    top_k_sampling, 
    top_p_sampling, 
    apply_repetition_penalty, 
    sample_from_logits
)
from model.transformer import PoetryGPT


class TestTopKSampling:
    def test_top_k_basic(self):
        """Test basic top-k sampling functionality"""
        # Create logits where we know the top-k
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])  # Top-3: [5.0, 4.0, 3.0]
        
        filtered = top_k_sampling(logits, k=3)
        
        # Only top-3 should remain, others should be -inf
        expected_mask = torch.tensor([[-float('inf'), 5.0, 3.0, -float('inf'), 4.0]])
        assert torch.equal(filtered, expected_mask)
    
    def test_top_k_disabled(self):
        """Test top-k with k=0 (disabled)"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        filtered = top_k_sampling(logits, k=0)
        
        # Should return original logits unchanged
        assert torch.equal(filtered, logits)
    
    def test_top_k_larger_than_vocab(self):
        """Test top-k with k larger than vocabulary size"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        filtered = top_k_sampling(logits, k=10)
        
        # Should return all tokens (no filtering)
        assert torch.equal(filtered, logits)


class TestTopPSampling:
    def test_top_p_basic(self):
        """Test basic top-p sampling functionality"""
        # Create logits that will have known probabilities
        logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])  # After softmax: ~[0.57, 0.21, 0.14, 0.05]
        
        filtered = top_p_sampling(logits, p=0.8)
        
        # Should keep tokens until cumulative prob > 0.8
        # Check that highest probability token is kept
        assert filtered[0, 0] == 2.0  # Keep highest
        
        # Check that at least some tokens are filtered out
        num_filtered = (filtered[0] == -float('inf')).sum().item()
        assert num_filtered > 0  # Should filter out some tokens
    
    def test_top_p_disabled(self):
        """Test top-p with p=1.0 (disabled)"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        filtered = top_p_sampling(logits, p=1.0)
        
        # Should return original logits
        assert torch.equal(filtered, logits)
    
    def test_top_p_keeps_at_least_one(self):
        """Test that top-p always keeps at least one token"""
        logits = torch.tensor([[10.0, 1.0, 1.0]])  # First token has very high prob
        
        filtered = top_p_sampling(logits, p=0.1)  # Very low threshold
        
        # Should still keep the first token
        assert filtered[0, 0] == 10.0
        assert filtered[0, 1] == -float('inf')
        assert filtered[0, 2] == -float('inf')


class TestRepetitionPenalty:
    def test_repetition_penalty_basic(self):
        """Test basic repetition penalty functionality"""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        input_ids = torch.tensor([[1, 3]])  # Tokens 1 and 3 appeared before
        
        penalized = apply_repetition_penalty(logits, input_ids, penalty=2.0)
        
        # Token 1 (positive logit): 2.0 / 2.0 = 1.0
        # Token 3 (positive logit): 4.0 / 2.0 = 2.0
        # Tokens 0, 2 should be unchanged
        assert penalized[0, 0] == 1.0  # unchanged
        assert penalized[0, 1] == 1.0  # penalized: 2.0 / 2.0
        assert penalized[0, 2] == 3.0  # unchanged  
        assert penalized[0, 3] == 2.0  # penalized: 4.0 / 2.0
    
    def test_repetition_penalty_negative_logits(self):
        """Test repetition penalty with negative logits"""
        logits = torch.tensor([[-1.0, -2.0, 1.0]])
        input_ids = torch.tensor([[0, 1]])  # Tokens 0 and 1 appeared before
        
        penalized = apply_repetition_penalty(logits, input_ids, penalty=2.0)
        
        # Negative logits get multiplied by penalty (become more negative)
        assert penalized[0, 0] == -2.0  # -1.0 * 2.0
        assert penalized[0, 1] == -4.0  # -2.0 * 2.0
        assert penalized[0, 2] == 1.0   # unchanged
    
    def test_repetition_penalty_disabled(self):
        """Test repetition penalty with penalty=1.0 (disabled)"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        input_ids = torch.tensor([[0, 1, 2]])
        
        penalized = apply_repetition_penalty(logits, input_ids, penalty=1.0)
        
        # Should be unchanged
        assert torch.equal(penalized, logits)
    
    def test_repetition_penalty_out_of_bounds_tokens(self):
        """Test repetition penalty handles out-of-bounds token IDs gracefully"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        input_ids = torch.tensor([[0, 999]])  # Token 999 is out of bounds
        
        # Should not crash
        penalized = apply_repetition_penalty(logits, input_ids, penalty=2.0)
        
        # Token 0 should be penalized, others unchanged
        assert penalized[0, 0] == 0.5  # 1.0 / 2.0
        assert penalized[0, 1] == 2.0  # unchanged
        assert penalized[0, 2] == 3.0  # unchanged


class TestSampleFromLogits:
    def test_sample_from_logits_basic(self):
        """Test basic sampling functionality"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        # Sample multiple times to test randomness
        samples = []
        for _ in range(100):
            sample = sample_from_logits(logits, temperature=1.0)
            samples.append(sample.item())
        
        # Should sample from all available tokens
        unique_samples = set(samples)
        assert len(unique_samples) > 1  # Should have some variety
        assert all(0 <= s <= 2 for s in unique_samples)  # All valid token IDs
    
    def test_sample_from_logits_with_temperature(self):
        """Test sampling with different temperatures"""
        logits = torch.tensor([[1.0, 10.0]])  # Second token much more likely
        
        # High temperature (more random)
        high_temp_samples = []
        for _ in range(100):
            sample = sample_from_logits(logits, temperature=2.0)
            high_temp_samples.append(sample.item())
        
        # Low temperature (more deterministic)
        low_temp_samples = []
        for _ in range(100):
            sample = sample_from_logits(logits, temperature=0.1)
            low_temp_samples.append(sample.item())
        
        # Low temperature should be more biased toward token 1
        high_temp_token1_ratio = sum(s == 1 for s in high_temp_samples) / 100
        low_temp_token1_ratio = sum(s == 1 for s in low_temp_samples) / 100
        
        assert low_temp_token1_ratio > high_temp_token1_ratio
    
    def test_sample_from_logits_with_top_k(self):
        """Test sampling with top-k filtering"""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        
        samples = []
        for _ in range(100):
            sample = sample_from_logits(logits, top_k=2)  # Only top 2 tokens
            samples.append(sample.item())
        
        unique_samples = set(samples)
        # Should only sample from tokens 3 and 4 (top 2)
        assert unique_samples.issubset({3, 4})
    
    def test_sample_from_logits_with_top_p(self):
        """Test sampling with top-p filtering"""
        logits = torch.tensor([[5.0, 1.0, 1.0, 1.0]])  # First token dominates
        
        samples = []
        for _ in range(100):
            sample = sample_from_logits(logits, top_p=0.8)
            samples.append(sample.item())
        
        # Should heavily favor token 0 due to nucleus sampling
        token_0_ratio = sum(s == 0 for s in samples) / 100
        assert token_0_ratio > 0.7  # Should be heavily biased
    
    def test_sample_from_logits_with_repetition_penalty(self):
        """Test sampling with repetition penalty"""
        logits = torch.tensor([[1.0, 1.0, 1.0]])  # Equal probabilities
        input_ids = torch.tensor([[0, 0, 1]])  # Tokens 0 and 1 repeated
        
        samples = []
        for _ in range(100):
            sample = sample_from_logits(
                logits, 
                input_ids=input_ids, 
                repetition_penalty=5.0  # Strong penalty
            )
            samples.append(sample.item())
        
        # Should strongly favor token 2 (not repeated)
        token_2_ratio = sum(s == 2 for s in samples) / 100
        assert token_2_ratio > 0.4  # Should be biased toward non-repeated token


class TestModelGenerateWithImprovedSampling:
    def test_model_generate_with_top_p(self):
        """Test model generation with top-p sampling"""
        model = PoetryGPT(vocab_size=50, d_model=32, n_heads=4, n_layers=2)
        input_ids = torch.randint(0, 50, (1, 5))
        
        generated = model.generate(
            input_ids, 
            max_new_tokens=10, 
            top_p=0.9
        )
        
        assert generated.shape == (1, 15)  # 5 + 10 tokens
        assert torch.all(generated >= 0)
        assert torch.all(generated < 50)
    
    def test_model_generate_with_repetition_penalty(self):
        """Test model generation with repetition penalty"""
        model = PoetryGPT(vocab_size=50, d_model=32, n_heads=4, n_layers=2)
        input_ids = torch.randint(0, 50, (1, 5))
        
        generated = model.generate(
            input_ids, 
            max_new_tokens=10, 
            repetition_penalty=2.0
        )
        
        assert generated.shape == (1, 15)
        assert torch.all(generated >= 0) 
        assert torch.all(generated < 50)
    
    def test_model_generate_with_all_options(self):
        """Test model generation with all sampling options"""
        model = PoetryGPT(vocab_size=50, d_model=32, n_heads=4, n_layers=2)
        input_ids = torch.randint(0, 50, (1, 5))
        
        generated = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.8,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        assert generated.shape == (1, 15)
        assert torch.all(generated >= 0)
        assert torch.all(generated < 50)
        
        # Check that we start with the input
        assert torch.equal(generated[:, :5], input_ids)