"""
HFOLD Attention Implementation: Dynamic Hidden State Heap Folding for Efficient Attention

.. deprecated::
    This module (`hfold_attention`) is superseded by :mod:`hfold.core.hfold_attention_v2`,
    which matches the HFold proposal (joint softmax, heap pop correctness, folding).
    Prefer importing ``HFoldMultiHeadAttention`` from ``hfold.core.hfold_attention_v2``.

Core algorithm:
- Maintains a max-heap of size s (constant)
- At each timestep, processes k tokens in sliding window
- Adds top q attention-scoring keys to heap
- Pops top e keys and attends to them
- Folds removed tokens into remaining heap nodes via gated linear combination
- Results in O(n) complexity while preserving long-range context via "folded hidden states"
"""

import math
import heapq
import warnings
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.warn(
    "hfold.core.hfold_attention is deprecated; use hfold.core.hfold_attention_v2 "
    "(HFoldMultiHeadAttention, HFoldAttentionV2).",
    DeprecationWarning,
    stacklevel=1,
)


class HeapNode:
    """Represents a node in the HFOLD heap"""
    
    def __init__(self, attention_score: float, key: torch.Tensor, index: int):
        """
        Args:
            attention_score: Score for ordering in max-heap (negated for min-heap simulation)
            key: Key vector (d_k dimension)
            index: Unique index for stable comparison
        """
        self.attention_score = attention_score
        self.key = key
        self.index = index
        # Accumulated folded state - weighted combination of removed tokens
        self.folded_state = None
    
    def __lt__(self, other):
        """For heap comparison (Python uses min-heap by default)"""
        # Negate for max-heap behavior
        if -self.attention_score != -other.attention_score:
            return -self.attention_score < -other.attention_score
        return self.index < other.index
    
    def __repr__(self):
        return f"HeapNode(score={self.attention_score:.4f}, idx={self.index})"


class SlidingWindowAttention(nn.Module):
    """Standard sliding window attention mechanism"""
    
    def __init__(self, window_size: int, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    
    def forward(
        self,
        Q: torch.Tensor,  # (batch, heads, seq_len, d_k)
        K: torch.Tensor,  # (batch, heads, seq_len, d_k)
        V: torch.Tensor,  # (batch, heads, seq_len, d_k)
        token_idx: int,   # Current token position
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sliding window attention for current token
        
        Args:
            Q: Query for current token (batch, heads, 1, d_k)
            K: Keys for sliding window (batch, heads, window_size, d_k)
            V: Values for sliding window (batch, heads, window_size, d_k)
            token_idx: Current token position (for masking)
        
        Returns:
            output: Attention output (batch, heads, 1, d_k)
            attention_weights: Attention scores (batch, heads, 1, window_size)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, heads, 1, window_size)
        
        # Apply causal mask if needed (ensure only past tokens)
        # For sliding window, all tokens in window are valid
        attention_weights = F.softmax(scores, dim=-1)  # (batch, heads, 1, window_size)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # (batch, heads, 1, d_k)
        
        return output, attention_weights


class HFoldAttention(nn.Module):
    """
    HFOLD Attention Layer: Integrates sliding window with dynamic heap of folded hidden states
    
    Algorithm:
    1. Compute sliding window attention for k tokens
    2. Extract top q attention scores (keys)
    3. Add these keys to max-heap (size s)
    4. Pop top e keys from heap
    5. Attend to (k + e) tokens total
    6. Fold removed tokens into remaining heap nodes
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,           # k: sliding window size
        heap_size: int,              # s: max heap size (constant)
        q_topk: int,                 # q: top-k keys to add to heap
        e_pop: int,                  # e: top keys to pop from heap
        dropout: float = 0.1,
        fold_nonlinearity: str = "gelu",  # Gating mechanism for folding
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            window_size: Size of sliding window (k)
            heap_size: Maximum heap size (s)
            q_topk: Number of top keys to add to heap (q)
            e_pop: Number of keys to pop from heap (e)
            dropout: Dropout rate
            fold_nonlinearity: Nonlinearity for gated combination during folding
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        self.heap_size = heap_size
        self.q_topk = q_topk
        self.e_pop = e_pop
        self.dropout = nn.Dropout(dropout)
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert window_size > 0, "window_size must be positive"
        assert e_pop <= heap_size, "e_pop cannot exceed heap_size"
        assert q_topk > 0, "q_topk must be positive"
        
        # Sliding window attention component
        self.window_attention = SlidingWindowAttention(
            window_size, d_model, n_heads, dropout
        )
        
        # Linear projections for heap keys
        # When we retrieve from heap, we need new Q,K,V projections for them
        self.heap_key_proj = nn.Linear(d_model, d_model)
        self.heap_value_proj = nn.Linear(d_model, d_model)
        
        # Gated folding mechanism: for combining removed tokens into heap nodes
        # Linear transformation for folding: combines removed tokens into hidden state
        self.fold_gate_proj = nn.Linear(d_model, d_model)
        self.fold_value_proj = nn.Linear(d_model, d_model)
        
        if fold_nonlinearity == "gelu":
            self.fold_nonlinearity = F.gelu
        elif fold_nonlinearity == "relu":
            self.fold_nonlinearity = F.relu
        else:
            raise ValueError(f"Unknown nonlinearity: {fold_nonlinearity}")
    
    def forward(
        self,
        Q: torch.Tensor,              # (batch, heads, 1, d_k) - current token query
        K_full: torch.Tensor,          # (batch, heads, seq_len, d_k) - all keys so far
        V_full: torch.Tensor,          # (batch, heads, seq_len, d_k) - all values so far
        token_idx: int,
        heap: Optional[List] = None,  # Heap state to maintain across tokens
    ) -> Tuple[torch.Tensor, Optional[List], dict]:
        """
        HFOLD attention forward pass
        
        Args:
            Q: Current token query (batch, heads, 1, d_k)
            K_full: All historical keys (batch, heads, seq_len, d_k)
            V_full: All historical values (batch, heads, seq_len, d_k)
            token_idx: Current token index
            heap: Heap state from previous token (list of HeapNodes)
        
        Returns:
            output: Attention output (batch, heads, 1, d_k)
            heap: Updated heap for next token
            debug_info: Dictionary with algorithm metrics
        """
        batch_size, n_heads, _, d_k = Q.shape
        
        if heap is None:
            heap = []
        
        debug_info = {
            'window_attention_scores': None,
            'heap_size': len(heap),
            'keys_added_to_heap': 0,
            'keys_removed_from_heap': 0,
        }
        
        # Step 1: Get sliding window keys/values (last k tokens, or all if seq_len < k)
        start_idx = max(0, token_idx - self.window_size)
        K_window = K_full[:, :, start_idx:token_idx + 1, :]
        V_window = V_full[:, :, start_idx:token_idx + 1, :]
        
        # Step 2: Compute sliding window attention
        window_output, window_scores = self.window_attention(Q, K_window, V_window, token_idx)
        # window_scores: (batch, heads, 1, window_size)
        
        # Step 3: Extract top q keys from sliding window (based on attention scores)
        # Flatten batch and heads for easier processing
        window_scores_flat = window_scores.squeeze(2)  # (batch, heads, window_size)
        
        # For each head, extract top-q attention scores
        top_q_values, top_q_indices = torch.topk(
            window_scores_flat, k=min(self.q_topk, window_scores_flat.size(-1)), dim=-1
        )  # (batch, heads, q_topk)
        
        # Convert indices back to full sequence indices
        top_q_key_indices = top_q_indices + start_idx  # Correct for window offset
        
        # Step 4: Add top-q keys to heap
        # We process this per sample and per head
        for b in range(batch_size):
            for h in range(n_heads):
                for idx_pos in range(top_q_indices.shape[-1]):
                    key_idx = top_q_key_indices[b, h, idx_pos].item()
                    attn_score = top_q_values[b, h, idx_pos].item()
                    
                    # Extract the actual key vector
                    key_vector = K_full[b, h, key_idx, :].detach()  # (d_k,)
                    
                    # Create heap node
                    node = HeapNode(attn_score, key_vector, key_idx)
                    
                    # Add to heap (min-heap, with negated scores for max-heap)
                    if len(heap) < self.heap_size:
                        heapq.heappush(heap, node)
                        debug_info['keys_added_to_heap'] += 1
                    elif attn_score > heap[0].attention_score:
                        # Pop lowest score and add new one
                        heapq.heapreplace(heap, node)
                        debug_info['keys_added_to_heap'] += 1
                        debug_info['keys_removed_from_heap'] += 1
        
        # Step 5: Pop top e keys from heap and attend to them
        keys_to_pop = min(self.e_pop, len(heap))
        popped_nodes = []
        
        if keys_to_pop > 0:
            # Extract top keys without destroying heap (use heapq.nlargest)
            popped_nodes = heapq.nlargest(keys_to_pop, heap)
            
            # Remove these from heap
            for _ in range(keys_to_pop):
                if len(heap) > 0:
                    heapq.heappop(heap)
            
            debug_info['keys_removed_from_heap'] += keys_to_pop
        
        # Step 6: Attend to heap keys + sliding window keys
        # Combine window and heap attention
        heap_output = window_output
        
        if len(popped_nodes) > 0:
            # Create K,V from popped heap nodes
            popped_keys = torch.stack([node.key for node in popped_nodes], dim=0)  # (e_pop, d_k)
            
            # Project heap keys/values
            # For each head, recompute projections
            K_heap = self.heap_key_proj(popped_keys).unsqueeze(0).unsqueeze(0)  # (1, 1, e_pop, d_model)
            K_heap = K_heap.view(1, 1, -1, self.d_k)  # Reshape to (1, 1, e_pop, d_k)
            
            # For values, we need to get them from V_full using the key indices
            heap_value_indices = [node.index for node in popped_nodes]
            V_heap = V_full[0, :, heap_value_indices, :]  # (heads, e_pop, d_k)
            V_heap = V_heap.unsqueeze(0)  # (1, heads, e_pop, d_k)
            
            # Compute attention scores for heap keys
            heap_scores = torch.matmul(Q, K_heap.transpose(-2, -1)) / math.sqrt(self.d_k)
            heap_attention = F.softmax(heap_scores, dim=-1)
            heap_attention = self.dropout(heap_attention)
            
            # Apply to heap values
            heap_attention_output = torch.matmul(heap_attention, V_heap)
            
            # Combine window and heap attention
            total_attention_vectors = 1 + keys_to_pop  # Current window + popped heap keys
            window_attn_weight = (self.window_size + len(window_scores_flat)) / total_attention_vectors
            heap_attn_weight = keys_to_pop / total_attention_vectors
            
            heap_output = window_attn_weight * window_output + heap_attn_weight * heap_attention_output
        
        # Step 7: Fold removed tokens into remaining heap nodes
        if keys_to_pop > 0 and len(heap) > 0:
            removed_tokens = []
            
            # Collect vectors that were removed (both from window and from heap pops)
            # For simplicity, we'll add folded information to remaining heap nodes
            for node in heap:
                # Each remaining node gets a folded state that encodes removed tokens
                # This is implemented as a gated combination with learnable transformation
                
                if len(removed_tokens) == 0:
                    # Initialize folded state as zero the first time
                    node.folded_state = torch.zeros_like(node.key)
                else:
                    # Create a learnable combination of removed tokens
                    removed_stack = torch.stack(removed_tokens, dim=0)  # (n_removed, d_k)
                    
                    # Gated combination: fold_gate * removed_values + (1 - fold_gate) * existing_folded
                    gate = torch.sigmoid(self.fold_gate_proj(removed_stack.mean(dim=0)))
                    fold_delta = self.fold_nonlinearity(self.fold_value_proj(removed_stack.mean(dim=0)))
                    
                    if node.folded_state is None:
                        node.folded_state = gate * fold_delta
                    else:
                        node.folded_state = (1 - gate) * node.folded_state + gate * fold_delta
        
        debug_info['heap_size'] = len(heap)
        
        return heap_output, heap, debug_info


class HFoldMultiHeadAttention(nn.Module):
    """Multi-head HFOLD attention with proper Q,K,V projections"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        heap_size: int,
        q_topk: int,
        e_pop: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Input projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # HFOLD attention for each head
        self.hfold_attn = HFoldAttention(
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            heap_size=heap_size,
            q_topk=q_topk,
            e_pop=e_pop,
            dropout=dropout,
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        Q: torch.Tensor,              # (batch, seq_len, d_model)
        K: torch.Tensor,              # (batch, seq_len, d_model)
        V: torch.Tensor,              # (batch, seq_len, d_model)
        token_idx: int,
        heap: Optional[List] = None,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Multi-head HFOLD attention
        
        Args:
            Q, K, V: (batch, seq_len, d_model)
            token_idx: Current token position
            heap: Heap state
        
        Returns:
            output: (batch, 1, d_model)
            heap: Updated heap
        """
        batch_size = Q.shape[0]
        
        # Project to multi-head
        Q_projected = self.W_q(Q[:, -1:, :])  # (batch, 1, d_model)
        K_projected = self.W_k(K)              # (batch, seq_len, d_model)
        V_projected = self.W_v(V)              # (batch, seq_len, d_model)
        
        # Reshape to (batch, n_heads, seq_len, d_k)
        Q_heads = Q_projected.view(batch_size, 1, self.n_heads, self.d_k).transpose(1, 2)
        K_heads = K_projected.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V_heads = V_projected.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply HFOLD attention
        attn_output, heap, _ = self.hfold_attn(
            Q_heads, K_heads, V_heads, token_idx, heap
        )
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, n_heads, 1, d_k)
        attn_output = attn_output.view(batch_size, 1, self.d_model)
        
        # Output projection
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        return output, heap
