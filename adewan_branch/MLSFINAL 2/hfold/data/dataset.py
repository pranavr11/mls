"""
Data loading utilities for PG-19 and other language modeling datasets
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class TokenizedDataset(torch.utils.data.Dataset):
    """Dataset of pre-tokenized sequences"""
    
    def __init__(
        self,
        token_ids: torch.Tensor,
        block_size: int,
        stride: int = 1,
    ):
        """
        Args:
            token_ids: Tensor of shape (total_tokens,) with all token IDs
            block_size: Context length for each example
            stride: How many tokens to skip between examples (for overlapping windows)
        """
        self.token_ids = token_ids
        self.block_size = block_size
        self.stride = stride
        
        # Calculate valid starting positions
        valid_starts = []
        for start in range(0, len(token_ids) - block_size, stride):
            if start + block_size + 1 <= len(token_ids):
                valid_starts.append(start)
        
        self.valid_starts = valid_starts
    
    def __len__(self) -> int:
        return len(self.valid_starts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training example
        
        Returns:
            {
                'input_ids': (block_size,),
                'labels': (block_size,),
            }
        """
        start = self.valid_starts[idx]
        end = start + self.block_size
        
        chunk = self.token_ids[start:end + 1]
        
        return {
            'input_ids': chunk[:-1].long(),
            'labels': chunk[1:].long(),
        }


class TextDataset(torch.utils.data.Dataset):
    """Raw text dataset that tokenizes on-the-fly"""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        block_size: int = 1024,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer object with encode/decode methods
            block_size: Context length
            max_samples: Maximum number of samples to use
        """
        self.block_size = block_size
        self.tokenizer = tokenizer
        
        # Tokenize all texts
        all_tokens = []
        for text in texts[:max_samples] if max_samples else texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            all_tokens.append(tokenizer.eos_token_id)  # Add separator
        
        self.token_ids = torch.tensor(all_tokens, dtype=torch.long)
        
        # Create valid starting positions
        self.starts = []
        for i in range(0, len(self.token_ids) - block_size):
            self.starts.append(i)
    
    def __len__(self) -> int:
        return len(self.starts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.starts[idx]
        chunk = self.token_ids[start:start + self.block_size + 1]
        
        return {
            'input_ids': chunk[:-1],
            'labels': chunk[1:],
        }


class PG19Loader:
    """Load and process Project Gutenberg-19 dataset"""
    
    # Common markers in Project Gutenberg texts
    START_MARKERS = [
        "***START OF THIS PROJECT GUTENBERG EBOOK",
        "***START***",
    ]
    END_MARKERS = [
        "***END OF THIS PROJECT GUTENBERG EBOOK",
        "***END***",
    ]
    
    def __init__(self, data_path: str, tokenizer=None):
        """
        Args:
            data_path: Path to PG19 data directory
            tokenizer: Tokenizer to use (if None, uses simple word-level)
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Remove Project Gutenberg headers/footers"""
        
        # Find actual content boundaries
        start_idx = 0
        for marker in PG19Loader.START_MARKERS:
            idx = text.find(marker)
            if idx != -1:
                start_idx = max(start_idx, idx + len(marker))
        
        end_idx = len(text)
        for marker in PG19Loader.END_MARKERS:
            idx = text.rfind(marker)
            if idx != -1:
                end_idx = min(end_idx, idx)
        
        text = text[start_idx:end_idx]
        
        # Basic cleaning
        text = text.strip()
        # Remove multiple whitespace
        text = ' '.join(text.split())
        
        return text
    
    def load_texts(self, num_books: Optional[int] = None) -> List[str]:
        """Load PG19 texts"""
        
        texts = []
        
        # Check if data_path is a directory or single file
        if self.data_path.is_dir():
            files = list(self.data_path.glob("*.txt"))
            if num_books:
                files = files[:num_books]
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                        text = self.clean_text(text)
                        if text:  # Only add non-empty texts
                            texts.append(text)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return texts
    
    def create_dataset(
        self,
        block_size: int,
        num_books: int = 100,
        split: str = 'train',
    ) -> TokenizedDataset:
        """
        Create a tokenized dataset
        
        Args:
            block_size: Context length
            num_books: Number of books to load
            split: 'train', 'val', or 'test'
        
        Returns:
            TokenizedDataset ready for training
        """
        # Load texts
        texts = self.load_texts(num_books=num_books)
        
        if not texts:
            raise ValueError(f"No texts found in {self.data_path}")
        
        # Combine and tokenize
        combined_text = " ".join(texts)
        
        if self.tokenizer is not None:
            token_ids = self.tokenizer.encode(combined_text)
        else:
            # Simple word-level tokenization
            words = combined_text.split()
            vocab = {w: i for i, w in enumerate(set(words))}
            token_ids = [vocab[w] for w in words]
        
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # Split into train/val/test
        total_len = len(token_ids)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        
        if split == 'train':
            token_ids = token_ids[:train_len]
        elif split == 'val':
            token_ids = token_ids[train_len:train_len + val_len]
        else:  # test
            token_ids = token_ids[train_len + val_len:]
        
        return TokenizedDataset(token_ids, block_size)


class DataCollator:
    """Collate batch of examples"""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples
        
        Args:
            batch: List of examples (dicts with 'input_ids' and 'labels')
        
        Returns:
            Batched tensors
        """
        input_ids = torch.stack([ex['input_ids'] for ex in batch])
        labels = torch.stack([ex['labels'] for ex in batch])
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }
