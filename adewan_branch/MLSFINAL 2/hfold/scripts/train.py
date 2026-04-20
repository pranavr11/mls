"""
Training script for HFOLD Transformer

Trains on PG19 dataset and compares perplexity with baselines
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HFoldTrainer:
    """Trainer for HFOLD models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler,
        device: torch.device,
        config: dict,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_tokens = 0
        log_interval = self.config.get('log_interval', 100)
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, return_logits=True, return_heaps=True)
            logits = outputs['logits']
            
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config['vocab_size']),
                labels.view(-1),
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Metrics
            batch_tokens = input_ids.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            self.global_step += 1
            
            # Logging
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / total_tokens
                perplexity = math.exp(avg_loss)
                throughput = batch_tokens / (time.time() - time.time())
                
                logger.info(
                    f"Epoch step {batch_idx+1}/{len(self.train_dataloader)} | "
                    f"Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens': total_tokens,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0
        total_tokens = 0
        
        for batch in self.val_dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(input_ids, return_logits=True, return_heaps=False)
            logits = outputs['logits']
            
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config['vocab_size']),
                labels.view(-1),
            )
            
            batch_tokens = input_ids.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens': total_tokens,
        }
    
    def train(self, num_epochs: int):
        """Train for multiple epochs"""
        
        logger.info("=" * 60)
        logger.info("Starting HFOLD Training")
        logger.info("=" * 60)
        logger.info(f"Model parameters: {self.model.get_num_params():,}")
        logger.info(f"Trainable parameters: {self.model.get_num_trainable_params():,}")
        logger.info("=" * 60)
        
        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            logger.info(
                f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.1f}s\n"
                f"  Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Perplexity: {train_metrics['perplexity']:.2f}\n"
                f"  Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Perplexity: {val_metrics['perplexity']:.2f}\n"
            )
            
            # Save checkpoint if best
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(f"  ✓ Best model saved (val_loss: {val_metrics['loss']:.4f})")
            
            # Regular checkpoint
            if (epoch + 1) % self.config.get('save_interval', 5) == 0:
                self._save_checkpoint(epoch, val_metrics)
    
    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'global_step': self.global_step,
        }
        
        checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints'))
        checkpoint_dir.mkdir(exist_ok=True)
        
        if is_best:
            path = checkpoint_dir / 'best_model.pt'
        else:
            path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")


def create_trainer(
    config_dict: dict,
    device: torch.device,
) -> Tuple[HFoldTrainer, dict]:
    """Create trainer and necessary components"""
    
    # Import here to avoid circular imports
    from hfold.models.hfold_transformer import HFoldTransformer
    from hfold.core.config import HFoldConfig
    from hfold.data.dataset import DataCollator
    
    logger.info("Initializing model...")
    
    # Create model config
    model_config = HFoldConfig(
        vocab_size=config_dict['vocab_size'],
        d_model=config_dict.get('d_model', 768),
        n_heads=config_dict.get('n_heads', 12),
        n_layers=config_dict.get('n_layers', 12),
        d_ff=config_dict.get('d_ff', 3072),
        window_size=config_dict.get('window_size', 64),
        heap_size=config_dict.get('heap_size', 32),
        q_topk=config_dict.get('q_topk', 16),
        e_pop=config_dict.get('e_pop', 8),
        dropout=config_dict.get('dropout', 0.1),
    )
    
    # Create model
    model = HFoldTransformer(model_config)
    model = model.to(device)
    
    # Create optimizer
    optimizer_config = {
        'lr': config_dict.get('learning_rate', 5e-5),
        'weight_decay': config_dict.get('weight_decay', 0.01),
        'betas': (0.9, 0.95),
    }
    optimizer = optim.AdamW(model.parameters(), **optimizer_config)
    
    # Create scheduler
    total_steps = config_dict.get('total_steps', 100000)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    logger.info(f"Model created with {model.get_num_params():,} parameters")
    
    return model, optimizer, scheduler, model_config


def main():
    """Main training script"""
    
    parser = argparse.ArgumentParser(description='Train HFOLD Transformer')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--data-path', type=str, default='/tmp/pg19', help='Data path')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Ensure CUDA
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Default config
    config = {
        'vocab_size': 50257,  # GPT-2
        'd_model': 768,
        'n_heads': 12,
        'n_layers': 12,
        'd_ff': 3072,
        'window_size': 64,
        'heap_size': 32,
        'q_topk': 16,
        'e_pop': 8,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'checkpoint_dir': args.checkpoint_dir,
        'gradient_clip': 1.0,
        'dropout': 0.1,
    }
    
    logger.info("Training config:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")
    
    # Create trainer
    model, optimizer, scheduler, model_config = create_trainer(config, device)
    
    # Create dummy dataloaders (in practice, load from PG19)
    logger.info(f"Creating data loaders...")
    
    # Create simple dummy data for testing
    dummy_dataset = PreTrainedTokenizedDataset(seq_len=1024, num_samples=100, vocab_size=50257)
    collator = DataCollator()
    
    train_loader = DataLoader(
        dummy_dataset,
        batch_size=config['batch_size'],
        collate_fn=collator,
        shuffle=True,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        dummy_dataset,
        batch_size=config['batch_size'],
        collate_fn=collator,
        num_workers=0,
    )
    
    # Create trainer
    trainer = HFoldTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'])


class PreTrainedTokenizedDataset(torch.utils.data.Dataset):
    """Simple dataset for testing"""
    
    def __init__(self, seq_len: int = 1024, num_samples: int = 100, vocab_size: int = 50257):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {'input_ids': input_ids, 'labels': labels}


if __name__ == '__main__':
    main()
