import os
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.loader import create_poetry_dataloader
from .scheduler import get_lr_scheduler, clip_gradients


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


def train_gpt(model, config):
    """Train the GPT model using subword tokenization"""
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create data loader with subword tokenization
    dataloader, dataset = create_poetry_dataloader(
        vocab_size=config.vocab_size,
        seq_length=config.seq_len,
        batch_size=config.batch_size,
        tokenizer_name=config.tokenizer_name
    )
    
    # Calculate total training steps for scheduler
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * config.epochs
    
    # Initialize learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, config, total_steps)
    
    # Try to load existing checkpoint
    start_epoch = load_checkpoint(model, optimizer, config.checkpoint_path)
    
    model.train()
    
    # Start total training timer
    training_start_time = time.time()
    avg_loss = 0.0  # Initialize avg_loss
    global_step = start_epoch * steps_per_epoch  # Track global step for scheduler
    
    print(f"Training setup:")
    print(f"  - Steps per epoch: {steps_per_epoch}")
    print(f"  - Total steps: {total_steps}")
    print(f"  - Learning rate scheduler: {'enabled' if scheduler else 'disabled'}")
    print(f"  - Gradient clipping: {config.gradient_clip_val}")
    print(f"  - Weight decay: {config.weight_decay}")
    print()
    
    for epoch in range(start_epoch, config.epochs):
        # Start epoch timer
        epoch_start_time = time.time()
        total_loss = 0
        batch_count = 0
        
        # Create progress bar for batches
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.epochs}', leave=False)
        
        for batch in progress_bar:
            # Extract input_ids and labels from batch
            input_ids = batch['input_ids'].to(config.device)
            labels = batch['labels'].to(config.device)
                
            # Forward pass
            logits = model(input_ids)  # [batch_size, seq_len, vocab_size]
                
            # Calculate loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
                
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = clip_gradients(model, config.gradient_clip_val)
            
            optimizer.step()
            
            # Update learning rate scheduler
            if scheduler:
                scheduler.step()
                global_step += 1
            
            total_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch time and loss
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / max(batch_count, 1)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else config.learning_rate
        
        print(f"Epoch {epoch + 1}/{config.epochs} completed in {format_time(epoch_time)}, "
              f"Average Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
        
        # Save checkpoint periodically
        if (epoch + 1) % config.save_every_epochs == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, config.checkpoint_path)
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    print(f"\nTraining completed in {format_time(total_training_time)}")
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, config.epochs, avg_loss, config.checkpoint_path)
    
    return model