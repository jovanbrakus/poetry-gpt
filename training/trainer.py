import os
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.loader import create_data_loader


def format_time(seconds):
    """Format seconds into minutes and seconds"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}m {seconds}s"


def save_checkpoint(model, optimizer, epoch, loss, path="checkpoints/gpt_checkpoint.pt"):
    """Save training checkpoint"""
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(model, optimizer, path="checkpoints/gpt_checkpoint.pt"):
    """Load training checkpoint"""
    if path and os.path.exists(path):
        checkpoint = torch.load(path, map_location=next(model.parameters()).device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Resumed from epoch {epoch}, loss {loss:.4f}")
        return epoch
    return 0


def train_gpt(model, text_data, tokenizer, config):
    """Train the GPT model on text data"""
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Try to load existing checkpoint
    start_epoch = load_checkpoint(model, optimizer, config.checkpoint_path)
    
    # Create data loader
    get_batch = create_data_loader(text_data, tokenizer, config.batch_size, config.seq_len)
    
    model.train()
    
    # Start total training timer
    training_start_time = time.time()
    avg_loss = 0.0  # Initialize avg_loss
    
    for epoch in range(start_epoch, config.epochs):
        # Start epoch timer
        epoch_start_time = time.time()
        total_loss = 0
        batch_count = 0
        
        # Create progress bar for batches
        batch_generator = get_batch()
        
        try:
            for batch in batch_generator:
                batch = batch.to(config.device)
                
                # Input is all tokens except last, target is all tokens except first
                x = batch[:, :-1]  # [batch_size, seq_len]
                y = batch[:, 1:]   # [batch_size, seq_len]
                
                # Forward pass
                logits = model(x)  # [batch_size, seq_len, vocab_size]
                
                # Calculate loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
        except StopIteration:
            pass
        
        # Calculate epoch time and loss
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / max(batch_count, 1)
        print(f"Epoch {epoch + 1}/{config.epochs} completed in {format_time(epoch_time)}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % config.save_every_epochs == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, config.checkpoint_path)
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    print(f"\n✅ Training completed in {format_time(total_training_time)}")
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, config.epochs, avg_loss, config.checkpoint_path)
    
    return model