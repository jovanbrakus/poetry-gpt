import torch
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=query.dtype, device=query.device))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights