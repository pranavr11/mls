#!/usr/bin/env python3
"""Quick test of HFOLD implementation"""

import torch
import sys
sys.path.insert(0, '.')

print("Testing HFOLD implementation...")
print("=" * 70)

try:
    # Test 1: Config
    from hfold.core.config import HFoldConfig
    print('✓ Config imported')
    
    config = HFoldConfig(
        vocab_size=1000,
        d_model=256,
        n_heads=8,
        n_layers=2,
        window_size=16,
        heap_size=8,
        q_topk=4,
        e_pop=2,
    )
    print(f'✓ Config: {config.d_model}d, {config.n_heads} heads, window={config.window_size}, heap={config.heap_size}')
    
    # Test 2: Model
    from hfold.models.hfold_transformer import HFoldTransformer
    print('✓ Model imported')
    
    model = HFoldTransformer(config)
    print(f'✓ Model created: {model.get_num_params():,} parameters')
    
    # Test 3: Forward pass
    print("\n--- Testing forward pass ---")
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    print(f'Input shape: {input_ids.shape}')
    
    outputs = model(input_ids, return_logits=True, return_heaps=True)
    assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
    print(f'✓ Logits shape: {outputs["logits"].shape}')
    assert outputs['hidden_states'].shape == (batch_size, seq_len, config.d_model)
    print(f'✓ Hidden states shape: {outputs["hidden_states"].shape}')
    print(f'✓ Number of heaps: {len(outputs["heaps"])}')
    
    # Test 4: Gradient flow
    print("\n--- Testing gradient flow ---")
    loss = outputs['logits'].sum()
    loss.backward()
    num_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    print(f'✓ Gradients computed: {num_params_with_grad}/{total_params} params have gradients')
    
    # Test 5: Generation
    print("\n--- Testing generation ---")
    generated = model.generate(
        input_ids[:1, :8],
        max_new_tokens=16,
        temperature=0.8,
        top_k=50,
    )
    assert generated.shape[1] <= input_ids.shape[1] + 16
    print(f'✓ Generated shape: {generated.shape} (prompt_len=8 + new_tokens<=16)')
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
    
except Exception as e:
    import traceback
    print(f'\n✗ ERROR: {e}')
    traceback.print_exc()
    sys.exit(1)
