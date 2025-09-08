import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import scaled_dot_product_attention
from .embeddings import PositionalEncoding
from .rope import RoPEPositionalEncoding
from .utils import create_causal_mask
from .initialization import initialize_model
from .sampling import sample_from_logits
from .normalization import RMSNorm
from .activations import create_feedforward


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_rope=False, rope_max_len=2048):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Initialize RoPE if enabled
        if use_rope:
            self.rope = RoPEPositionalEncoding(self.d_k, max_len=rope_max_len)

    def forward(self, query, key, value, mask=None, position_offset=0):
        batch_size = query.size(0)
        seq_len = query.size(1)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape for multi-head attention: [batch, seq_len, num_heads, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Apply RoPE if enabled
        if self.use_rope:
            Q, K = self.rope(Q, K, seq_len=seq_len, offset=position_offset)

        # Reshape for attention computation: [batch * num_heads, seq_len, d_k]
        batch_heads = batch_size * self.num_heads
        Q = Q.reshape(batch_heads, seq_len, self.d_k)
        K = K.reshape(batch_heads, seq_len, self.d_k)
        V = V.reshape(batch_heads, seq_len, self.d_k)

        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        attn_output = attn_output.view(batch_size, self.num_heads, seq_len, self.d_k)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        output = self.W_o(attn_output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1, activation_type='swiglu', 
                 use_rope=False, rope_max_len=2048):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, use_rope=use_rope, 
                                          rope_max_len=rope_max_len)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.feed_forward = create_feedforward(d_model, d_ff, dropout, activation_type)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, position_offset=0):
        norm_1 = self.norm1(x)
        attn_output = self.attention(norm_1, norm_1, norm_1, mask, position_offset=position_offset)
        x = x + self.dropout(attn_output)

        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x


class PoetryGPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6,
                 d_ff=2048, max_len=1024, dropout=0.1, init_method='transformer',
                 activation_type='swiglu', use_rope=False):
        super().__init__()

        self.max_len = max_len
        self.use_rope = use_rope

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Only use absolute positional encoding if not using RoPE
        if not use_rope:
            self.position_encoding = PositionalEncoding(d_model, max_len)
        else:
            self.position_encoding = None

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, activation_type, 
                           use_rope=use_rope, rope_max_len=max_len)
            for _ in range(n_layers)
        ])

        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)
        
        # Apply weight initialization
        initialize_model(self, method=init_method)

    def forward(self, idx, mask=None, position_offset=0):
        batch_size, seq_len = idx.shape
        if mask is None:
            mask = create_causal_mask(seq_len, idx.device)

        out = self.token_embedding(idx)
        
        # Only apply absolute positional encoding if not using RoPE
        if not self.use_rope:
            out = self.position_encoding(out)
        
        out = self.dropout(out)

        for block in self.blocks:
            out = block(out, mask, position_offset=position_offset)

        out = self.ln_f(out)
        out = self.lm_head(out)

        return out

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, 
                top_p=None, repetition_penalty=1.0):
        """
        Generate new tokens autoregressively with improved sampling.
        
        Args:
            idx: [batch_size, seq_len] - starting context
            max_new_tokens: number of new tokens to generate
            temperature: higher = more random, lower = more deterministic
            top_k: only sample from top k tokens (None to disable)
            top_p: nucleus sampling threshold (None to disable)
            repetition_penalty: penalty for repeated tokens (1.0 = no penalty)
        
        Returns:
            Generated token sequence [batch_size, seq_len + max_new_tokens]
        """
        self.eval()

        for step in range(max_new_tokens):
            # Truncate to max context length
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len:]
            
            # Calculate position offset for RoPE (helps with longer sequences)
            position_offset = 0 if self.use_rope and idx.size(1) <= self.max_len else max(0, idx.size(1) - self.max_len)

            # Forward pass
            logits = self(idx_cond, position_offset=position_offset)
            
            # Get logits for next token prediction
            next_token_logits = logits[:, -1, :]

            # Sample using improved sampling methods
            idx_next = sample_from_logits(
                logits=next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                input_ids=idx,
                repetition_penalty=repetition_penalty
            )

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx