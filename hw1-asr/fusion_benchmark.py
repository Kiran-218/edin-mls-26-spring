import torch
import time
import sys
sys.path.insert(0, 'glm_asr_triton_template')
from layers import MLP

def benchmark_mlp(fused, runs=20):
    mlp = MLP(2048, 5632, activation="silu", use_gating=True)
    mlp.gate_proj.weight = torch.randn(5632, 2048, dtype=torch.float32)
    mlp.up_proj.weight = torch.randn(5632, 2048, dtype=torch.float32)
    mlp.down_proj.weight = torch.randn(2048, 5632, dtype=torch.float32)

    x = torch.randn(256, 2048, device='cuda', dtype=torch.float32)
    mlp.gate_proj.weight = mlp.gate_proj.weight.cuda()
    mlp.up_proj.weight = mlp.up_proj.weight.cuda()
    mlp.down_proj.weight = mlp.down_proj.weight.cuda()

    MLP.FUSED = fused

    # Warmup
    for _ in range(3):
        _ = mlp(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        _ = mlp(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / runs

    label = "Fused SwiGLU" if fused else "Unfused (standard)"
    print(f"{label:<25} {elapsed:>10.3f}ms")
    return elapsed

print("MLP Forward Pass Benchmark (hidden=2048, intermediate=5632)")
print(f"{'Config':<25} {'Time (ms)':>10}")
print("-" * 38)

unfused = benchmark_mlp(fused=False)
fused = benchmark_mlp(fused=True)

speedup = unfused / fused
print(f"\n✅ Fusion speedup: {speedup:.2f}x faster ({unfused:.3f}ms → {fused:.3f}ms)")
