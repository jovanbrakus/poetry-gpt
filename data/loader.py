import torch


def create_data_loader(text_data, tokenizer, batch_size=32, seq_len=128):
    """Create batches for training from text data"""
    data_indices = torch.tensor(tokenizer.encode(text_data), dtype=torch.long)
    
    def get_batch():
        """Generator that yields training batches"""
        for i in range(0, len(data_indices) - seq_len, batch_size * seq_len):
            batch = []
            for j in range(batch_size):
                if i + j * seq_len + seq_len + 1 < len(data_indices):
                    start_idx = i + j * seq_len
                    end_idx = start_idx + seq_len + 1
                    batch.append(data_indices[start_idx:end_idx])
            
            if batch:
                yield torch.stack(batch)
    
    return get_batch