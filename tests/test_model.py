import pytest
import torch
import torch.nn as nn
from model.transformer import PoetryGPT, TransformerBlock, MultiHeadAttention
from model.attention import scaled_dot_product_attention
from model.embeddings import PositionalEncoding
from model.utils import create_causal_mask


class TestScaledDotProductAttention:
    def test_attention_output_shape(self):
        batch_size, seq_len, d_k = 2, 10, 64
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)
        
        output, weights = scaled_dot_product_attention(Q, K, V)
        
        assert output.shape == (batch_size, seq_len, d_k)
        assert weights.shape == (batch_size, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self):
        batch_size, seq_len, d_k = 1, 5, 32
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)
        
        _, weights = scaled_dot_product_attention(Q, K, V)
        
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-6)
    
    def test_attention_with_mask(self):
        batch_size, seq_len, d_k = 1, 4, 32
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)
        
        mask = create_causal_mask(seq_len, torch.device('cpu'))
        _, weights = scaled_dot_product_attention(Q, K, V, mask)
        
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert weights[0, i, j].item() == 0.0


class TestPositionalEncoding:
    def test_positional_encoding_shape(self):
        d_model, max_len = 128, 100
        pe = PositionalEncoding(d_model, max_len)
        
        batch_size, seq_len = 2, 50
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pe(x)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_positional_encoding_different_positions(self):
        d_model, max_len = 64, 10
        pe = PositionalEncoding(d_model, max_len)
        
        x1 = torch.zeros(1, 1, d_model)
        x2 = torch.zeros(1, 1, d_model)
        
        out1 = pe(x1)
        out2 = pe(torch.cat([x1, x2], dim=1))[:, 1:2, :]
        
        assert not torch.allclose(out1, out2)


class TestCausalMask:
    def test_causal_mask_shape(self):
        seq_len = 5
        mask = create_causal_mask(seq_len, torch.device('cpu'))
        assert mask.shape == (seq_len, seq_len)
    
    def test_causal_mask_lower_triangular(self):
        seq_len = 4
        mask = create_causal_mask(seq_len, torch.device('cpu'))
        
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    assert mask[i, j] == False
                else:
                    assert mask[i, j] == True


class TestMultiHeadAttention:
    @pytest.fixture
    def attention_layer(self):
        return MultiHeadAttention(d_model=128, num_heads=8)
    
    def test_init_dimensions(self, attention_layer):
        assert attention_layer.d_model == 128
        assert attention_layer.num_heads == 8
        assert attention_layer.d_k == 16
    
    def test_init_invalid_dimensions(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=127, num_heads=8)
    
    def test_forward_shape(self, attention_layer):
        batch_size, seq_len, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = attention_layer(x, x, x)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_forward_with_mask(self, attention_layer):
        batch_size, seq_len, d_model = 1, 5, 128
        x = torch.randn(batch_size, seq_len, d_model)
        mask = create_causal_mask(seq_len, torch.device('cpu'))
        
        output = attention_layer(x, x, x, mask)
        assert output.shape == (batch_size, seq_len, d_model)


class TestTransformerBlock:
    @pytest.fixture
    def transformer_block(self):
        return TransformerBlock(d_model=128, num_heads=8, d_ff=512, dropout=0.1)
    
    def test_forward_shape(self, transformer_block):
        batch_size, seq_len, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = transformer_block(x)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_forward_with_mask(self, transformer_block):
        batch_size, seq_len, d_model = 1, 5, 128
        x = torch.randn(batch_size, seq_len, d_model)
        mask = create_causal_mask(seq_len, torch.device('cpu'))
        
        output = transformer_block(x, mask)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_residual_connections(self, transformer_block):
        transformer_block.eval()
        x = torch.randn(1, 5, 128)
        
        with torch.no_grad():
            output = transformer_block(x)
            
        assert not torch.allclose(x, output)


class TestPoetryGPT:
    @pytest.fixture
    def poetry_gpt(self):
        return PoetryGPT(
            vocab_size=100,
            d_model=128,
            n_heads=8,
            n_layers=4,
            d_ff=512,
            max_len=256,
            dropout=0.1
        )
    
    def test_init_parameters(self, poetry_gpt):
        assert poetry_gpt.max_len == 256
        assert len(poetry_gpt.blocks) == 4
        assert poetry_gpt.token_embedding.num_embeddings == 100
        assert poetry_gpt.token_embedding.embedding_dim == 128
        assert poetry_gpt.lm_head.out_features == 100
    
    def test_forward_shape(self, poetry_gpt):
        batch_size, seq_len = 2, 10
        vocab_size = 100
        idx = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        output = poetry_gpt(idx)
        assert output.shape == (batch_size, seq_len, vocab_size)
    
    def test_forward_with_mask(self, poetry_gpt):
        batch_size, seq_len = 1, 5
        vocab_size = 100
        idx = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = create_causal_mask(seq_len, torch.device('cpu'))
        
        output = poetry_gpt(idx, mask)
        assert output.shape == (batch_size, seq_len, vocab_size)
    
    def test_generate_basic(self, poetry_gpt):
        poetry_gpt.eval()
        batch_size, initial_len = 1, 5
        vocab_size = 100
        idx = torch.randint(0, vocab_size, (batch_size, initial_len))
        
        with torch.no_grad():
            generated = poetry_gpt.generate(idx, max_new_tokens=10)
        
        assert generated.shape == (batch_size, initial_len + 10)
        assert torch.all(generated[:, :initial_len] == idx)
    
    def test_generate_with_temperature(self, poetry_gpt):
        poetry_gpt.eval()
        idx = torch.randint(0, 100, (1, 5))
        
        with torch.no_grad():
            gen_low_temp = poetry_gpt.generate(idx, max_new_tokens=5, temperature=0.1)
            gen_high_temp = poetry_gpt.generate(idx, max_new_tokens=5, temperature=2.0)
        
        assert gen_low_temp.shape == gen_high_temp.shape
    
    def test_generate_with_top_k(self, poetry_gpt):
        poetry_gpt.eval()
        idx = torch.randint(0, 100, (1, 5))
        
        with torch.no_grad():
            generated = poetry_gpt.generate(idx, max_new_tokens=5, top_k=10)
        
        assert generated.shape == (1, 10)
    
    def test_generate_respects_max_len(self, poetry_gpt):
        poetry_gpt.eval()
        long_sequence_len = poetry_gpt.max_len + 50
        idx = torch.randint(0, 100, (1, long_sequence_len))
        
        with torch.no_grad():
            generated = poetry_gpt.generate(idx, max_new_tokens=5)
        
        assert generated.shape == (1, long_sequence_len + 5)
    
    def test_parameter_count(self, poetry_gpt):
        total_params = sum(p.numel() for p in poetry_gpt.parameters())
        assert total_params > 0
        
        trainable_params = sum(p.numel() for p in poetry_gpt.parameters() if p.requires_grad)
        assert trainable_params == total_params