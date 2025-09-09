import torch
from torch.utils.data import DataLoader
from .dataset import create_poetry_dataset, PoetryDataset


def create_data_loader(dataset: PoetryDataset, batch_size: int = 32, 
                      shuffle: bool = True, num_workers: int = 0):
    """
    Create PyTorch DataLoader for PoetryDataset.
    
    Args:
        dataset: PoetryDataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        
    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )


def create_poetry_dataloader(slovenian_path="static/slovenian", 
                           vocab_size=12000,
                           seq_length=256,
                           batch_size=32,
                           stride=None,
                           tokenizer_name="poetry_tokenizer",
                           shuffle=True) -> DataLoader:
    """
    Convenience function to create complete poetry DataLoader.
    
    Args:
        slovenian_path: Path to Slovenian poetry files
        vocab_size: Subword vocabulary size
        seq_length: Maximum sequence length
        batch_size: Training batch size
        stride: Stride for overlapping sequences
        tokenizer_name: Name for tokenizer model files
        shuffle: Whether to shuffle training data
        
    Returns:
        PyTorch DataLoader ready for training
    """
    # Create dataset with subword tokenization
    dataset = create_poetry_dataset(
        slovenian_path=slovenian_path,
        vocab_size=vocab_size,
        seq_length=seq_length,
        stride=stride,
        tokenizer_name=tokenizer_name
    )
    
    # Create DataLoader
    dataloader = create_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    print(f"DataLoader created:")
    print(f"   Batch size: {batch_size}")
    print(f"   Total batches: {len(dataloader):,}")
    print(f"   Shuffle: {shuffle}")
    
    return dataloader, dataset