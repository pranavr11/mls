"""
Testing utilities for HFOLD correctness validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import sys


def compute_memory_for_n_tokens(n: int, hidden_size: int, n_heads: int) -> Dict[str, float]:
    """
    Compute memory requirements for different attention mechanisms
    
    Args:
        n: Sequence length
        hidden_size: Model dimension
        n_heads: Number of attention heads
    
    Returns:
        Memory in MB for different mechanisms
    """
    d_k = hidden_size // n_heads
    bytes_per_tensor = 4  # float32
    
    full_attention_qk = (n * n * d_k) * n_heads * bytes_per_tensor / (1024**2)
    sliding_window_qk = (256 * n * d_k) * n_heads * bytes_per_tensor / (1024**2)  # window=256
    
    return {
        'full_attention_mb': full_attention_qk,
        'sliding_window_mb': sliding_window_qk,
        'hfold_mb': sliding_window_qk * 0.5,  # Rough estimate
    }


def compute_flops_for_n_tokens(
    n: int,
    hidden_size: int,
    n_heads: int,
    window_size: int,
    e_pop: int,
) -> Dict[str, float]:
    """
    Compute FLOPs for different attention mechanisms
    
    Args:
        n: Sequence length
        hidden_size: Model dimension
        n_heads: Number of attention heads
        window_size: Sliding window size (k)
        e_pop: Heap pop size (e)
    
    Returns:
        Approximate FLOPs (operations in billions)
    """
    d_k = hidden_size // n_heads
    
    # Full attention: 2*n²*d_k*n_heads (multiply + softmax approx)
    full_flops = (2 * n * n * d_k * n_heads) / 1e9
    
    # Sliding window: 2*n*k*d_k*n_heads
    window_flops = (2 * n * window_size * d_k * n_heads) / 1e9
    
    # HFOLD: 2*n*(window_size + e_pop)*d_k*n_heads
    hfold_flops = (2 * n * (window_size + e_pop) * d_k * n_heads) / 1e9
    
    return {
        'full_attention_gflops': full_flops,
        'sliding_window_gflops': window_flops,
        'hfold_gflops': hfold_flops,
    }


class AttentionCorrectnessTester:
    """Test correctness of HFOLD implementation"""
    
    @staticmethod
    def test_basic_shape_correctness():
        """Test that HFOLD produces correct output shapes"""
        from hfold.models.hfold_transformer import HFoldTransformer
        from hfold.core.config import HFoldConfig
        
        config = HFoldConfig(
            vocab_size=1000,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            window_size=8,
            heap_size=4,
            q_topk=2,
            e_pop=1,
        )
        
        model = HFoldTransformer(config)
        
        # Create dummy input
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Forward pass
        outputs = model(input_ids, return_logits=True, return_heaps=True)
        
        assert outputs['logits'].shape == (batch_size, seq_len, 1000), \
            f"Expected logits shape {(batch_size, seq_len, 1000)}, got {outputs['logits'].shape}"
        
        assert outputs['hidden_states'].shape == (batch_size, seq_len, 64), \
            f"Expected hidden states shape {(batch_size, seq_len, 64)}, got {outputs['hidden_states'].shape}"
        
        assert len(outputs['heaps']) == config.n_layers, \
            f"Expected {config.n_layers} heaps, got {len(outputs['heaps'])}"
        
        print("✓ Shape correctness test passed")
        return True
    
    @staticmethod
    def test_gradient_flow():
        """Test that gradients flow correctly through HFOLD"""
        from hfold.models.hfold_transformer import HFoldTransformer
        from hfold.core.config import HFoldConfig
        
        config = HFoldConfig(
            vocab_size=1000,
            d_model=64,
            n_heads=4,
            n_layers=1,
            d_ff=256,
            window_size=8,
            heap_size=4,
            q_topk=2,
            e_pop=1,
        )
        
        model = HFoldTransformer(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create dummy input
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        target_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Forward pass
        outputs = model(input_ids, return_logits=True, return_heaps=True)
        logits = outputs['logits']
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            target_ids.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"✗ No gradient for {name}")
                return False
        
        # Update
        optimizer.step()
        
        print("✓ Gradient flow test passed")
        return True
    
    @staticmethod
    def test_heap_operation():
        """Min-heap on score keeps the top-s highest scores (root = smallest among retained)."""
        import heapq

        heap = []
        scores = [0.5, 0.7, 0.3, 0.9, 0.6]

        class TestNode:
            def __init__(self, score, idx):
                self.score = score
                self.idx = idx

            def __lt__(self, other):
                if self.score != other.score:
                    return self.score < other.score
                return self.idx < other.idx

            def __repr__(self):
                return f"Node({self.score:.2f})"

        for i, score in enumerate(scores):
            node = TestNode(score, i)
            if len(heap) < 3:
                heapq.heappush(heap, node)
            elif score > heap[0].score:
                heapq.heapreplace(heap, node)

        heap_scores = sorted([n.score for n in heap], reverse=True)
        expected_top3 = sorted(scores, reverse=True)[:3]

        assert heap_scores == expected_top3, f"Expected {expected_top3}, got {heap_scores}"

        print("✓ Heap operation test passed")
        return True
    
    @staticmethod
    def test_long_context_retention():
        """Test that HFOLD retains long-range information via heap folding"""
        from hfold.models.hfold_transformer import HFoldTransformer
        from hfold.core.config import HFoldConfig
        
        # q_topk > e_pop so heaps retain entries after insert-then-pop each step
        config = HFoldConfig(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=1,
            d_ff=256,
            window_size=8,
            heap_size=8,
            q_topk=4,
            e_pop=2,
            max_seq_len=256,
        )
        
        model = HFoldTransformer(config)
        model.eval()
        
        # Create a long sequence
        long_seq_len = 64
        input_ids = torch.randint(0, 100, (1, long_seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids, return_logits=False, return_heaps=True)
        
        heaps = outputs["heaps"]
        # heaps: [layer][batch][head] -> HeapHeadBucket
        max_bucket_size = 0
        for layer in heaps:
            for batch_row in layer:
                for bucket in batch_row:
                    max_bucket_size = max(max_bucket_size, len(bucket))

        assert max_bucket_size > 0, "No heap bucket retained entries after long sequence"

        print(f"✓ Long context retention test passed (max heap entries in any bucket: {max_bucket_size})")
        return True

    @staticmethod
    def test_clone_heaps_nested_structure():
        """_clone_heaps must preserve [layer][batch][head] nesting (each head is its own HeapHeadBucket)."""
        from hfold.models.hfold_transformer import HFoldTransformer
        from hfold.core.config import HFoldConfig
        from hfold.core.hfold_attention_v2 import HeapHeadBucket

        config = HFoldConfig(
            vocab_size=50,
            d_model=36,
            n_heads=3,
            n_layers=2,
            d_ff=64,
            window_size=6,
            heap_size=6,
            q_topk=3,
            e_pop=2,
        )
        model = HFoldTransformer(config)
        heaps = model._init_heaps(batch_size=2)
        assert isinstance(heaps[0][0][0], HeapHeadBucket)
        cloned = model._clone_heaps(heaps)
        assert len(cloned) == config.n_layers
        assert len(cloned[0]) == 2
        assert len(cloned[0][0]) == config.n_heads
        assert cloned[0][0][0] is not heaps[0][0][0]
        print("✓ Heap clone structure test passed")
        return True


def run_all_correctness_tests() -> bool:
    """Run all correctness tests"""
    
    tests = [
        AttentionCorrectnessTester.test_basic_shape_correctness,
        AttentionCorrectnessTester.test_gradient_flow,
        AttentionCorrectnessTester.test_heap_operation,
        AttentionCorrectnessTester.test_long_context_retention,
        AttentionCorrectnessTester.test_clone_heaps_nested_structure,
    ]
    
    passed = 0
    failed = 0
    
    print("=" * 60)
    print("HFOLD CORRECTNESS TEST SUITE")
    print("=" * 60)
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with exception: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_correctness_tests()
    sys.exit(0 if success else 1)
