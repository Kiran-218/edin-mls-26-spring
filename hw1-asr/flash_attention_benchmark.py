import torch
import triton
import time
import sys
sys.path.insert(0, 'glm_asr_triton_template')
from attention import scaled_dot_product_attention, flash_attention_kernel, attention_scores_kernel, softmax_inplace_kernel, attention_output_kernel
import numpy as np

def benchmark_attention(seq_len, head_dim, num_heads=4, batch=2, runs=20):
    q = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    v = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)

    print(f"\nseq_len={seq_len}, head_dim={head_dim}, heads={num_heads}")
    print(f"{'Method':<25} {'Time (ms)':>12}")
    print("-" * 40)

    # Flash attention
    for _ in range(3):
        _ = scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        _ = scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    flash_time = (time.perf_counter() - start) * 1000 / runs
    print(f"{'Flash Attention':<25} {flash_time:>12.3f}")

    # Standard torch
    for _ in range(3):
        scores = torch.einsum("bnqd,bnkd->bnqk", q, k) / np.sqrt(head_dim)
        attn = torch.softmax(scores, dim=-1)
        _ = torch.einsum("bnqk,bnkd->bnqd", attn, v)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        scores = torch.einsum("bnqd,bnkd->bnqk", q, k) / np.sqrt(head_dim)
        attn = torch.softmax(scores, dim=-1)
        _ = torch.einsum("bnqk,bnkd->bnqd", attn, v)
    torch.cuda.synchronize()
    std_time = (time.perf_counter() - start) * 1000 / runs
    print(f"{'Standard (torch)':<25} {std_time:>12.3f}")

    speedup = std_time / flash_time
    print(f"\n✅ Flash speedup: {speedup:.2f}x")

# Test different sequence lengths
benchmark_attention(seq_len=64, head_dim=64)
benchmark_attention(seq_len=128, head_dim=64)
benchmark_attention(seq_len=256, head_dim=64)
benchmark_attention(seq_len=512, head_dim=64)
