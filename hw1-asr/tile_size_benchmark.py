import torch
import triton
import time
import sys
sys.path.insert(0, 'glm_asr_triton_template')
from layers import linear_kernel_tf32, pad_to_multiple

def benchmark_tile_sizes(M, K, N, configs, runs=10):
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)

    print(f"\nMatrix size: ({M}, {K}) @ ({K}, {N})")
    print(f"{'Config':<30} {'Time (ms)':>12}")
    print("-" * 45)

    best_time = float('inf')
    best_config = None

    for (TILE_M, TILE_N, TILE_K) in configs:
        try:
            M_pad = pad_to_multiple(M, TILE_M)
            N_pad = pad_to_multiple(N, TILE_N)
            K_pad = pad_to_multiple(K, TILE_K)

            a_pad = torch.zeros(M_pad, K_pad, device='cuda', dtype=torch.float32)
            b_pad = torch.zeros(K_pad, N_pad, device='cuda', dtype=torch.float32)
            a_pad[:M, :K] = a
            b_pad[:K, :N] = b
            c = torch.zeros(M_pad, N_pad, device='cuda', dtype=torch.float32)

            grid = (triton.cdiv(M_pad, TILE_M), triton.cdiv(N_pad, TILE_N))

            for _ in range(3):
                linear_kernel_tf32[grid](
                    a_pad, b_pad, c, M_pad, N_pad, K_pad,
                    a_pad.stride(0), a_pad.stride(1),
                    b_pad.stride(0), b_pad.stride(1),
                    c.stride(0), c.stride(1),
                    BLOCK_M=TILE_M, BLOCK_N=TILE_N, BLOCK_K=TILE_K,
                )
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(runs):
                linear_kernel_tf32[grid](
                    a_pad, b_pad, c, M_pad, N_pad, K_pad,
                    a_pad.stride(0), a_pad.stride(1),
                    b_pad.stride(0), b_pad.stride(1),
                    c.stride(0), c.stride(1),
                    BLOCK_M=TILE_M, BLOCK_N=TILE_N, BLOCK_K=TILE_K,
                )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000 / runs

            label = f"TILE_M={TILE_M}, TILE_N={TILE_N}, TILE_K={TILE_K}"
            print(f"{label:<30} {elapsed:>12.3f}")

            if elapsed < best_time:
                best_time = elapsed
                best_config = (TILE_M, TILE_N, TILE_K)

        except Exception as e:
            label = f"TILE_M={TILE_M}, TILE_N={TILE_N}, TILE_K={TILE_K}"
            print(f"{label:<30} {'FAILED':>12}")

    print(f"\n✅ Best tile config: TILE_M={best_config[0]}, TILE_N={best_config[1]}, TILE_K={best_config[2]} at {best_time:.3f}ms")
    return best_config


def benchmark_warps_stages(M, K, N, TILE_M, TILE_N, TILE_K, warp_stage_configs, runs=10):
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)

    M_pad = pad_to_multiple(M, TILE_M)
    N_pad = pad_to_multiple(N, TILE_N)
    K_pad = pad_to_multiple(K, TILE_K)

    a_pad = torch.zeros(M_pad, K_pad, device='cuda', dtype=torch.float32)
    b_pad = torch.zeros(K_pad, N_pad, device='cuda', dtype=torch.float32)
    a_pad[:M, :K] = a
    b_pad[:K, :N] = b
    c = torch.zeros(M_pad, N_pad, device='cuda', dtype=torch.float32)

    grid = (triton.cdiv(M_pad, TILE_M), triton.cdiv(N_pad, TILE_N))

    print(f"\nWarps & Stages tuning — Matrix: ({M}, {K}) @ ({K}, {N})")
    print(f"Fixed tile: TILE_M={TILE_M}, TILE_N={TILE_N}, TILE_K={TILE_K}")
    print(f"{'num_warps':<15} {'num_stages':<15} {'Time (ms)':>12}")
    print("-" * 45)

    best_time = float('inf')
    best_config = None

    for (num_warps, num_stages) in warp_stage_configs:
        try:
            for _ in range(3):
                linear_kernel_tf32[grid](
                    a_pad, b_pad, c, M_pad, N_pad, K_pad,
                    a_pad.stride(0), a_pad.stride(1),
                    b_pad.stride(0), b_pad.stride(1),
                    c.stride(0), c.stride(1),
                    BLOCK_M=TILE_M, BLOCK_N=TILE_N, BLOCK_K=TILE_K,
                    num_warps=num_warps, num_stages=num_stages,
                )
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(runs):
                linear_kernel_tf32[grid](
                    a_pad, b_pad, c, M_pad, N_pad, K_pad,
                    a_pad.stride(0), a_pad.stride(1),
                    b_pad.stride(0), b_pad.stride(1),
                    c.stride(0), c.stride(1),
                    BLOCK_M=TILE_M, BLOCK_N=TILE_N, BLOCK_K=TILE_K,
                    num_warps=num_warps, num_stages=num_stages,
                )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000 / runs

            print(f"{num_warps:<15} {num_stages:<15} {elapsed:>12.3f}")

            if elapsed < best_time:
                best_time = elapsed
                best_config = (num_warps, num_stages)

        except Exception as e:
            print(f"{num_warps:<15} {num_stages:<15} {'FAILED':>12}")

    print(f"\n✅ Best warps/stages: num_warps={best_config[0]}, num_stages={best_config[1]} at {best_time:.3f}ms")
    return best_config


if __name__ == "__main__":
    # ── Part 1: Tile size tuning ──────────────────────────────────────
    print("=" * 50)
    print("PART 1: Tile Size Tuning")
    print("=" * 50)

    tile_configs = [
        (32, 32, 32),
        (64, 64, 32),
        (128, 64, 32),
        (64, 128, 32),
        (128, 128, 32),
        (64, 64, 64),
        (128, 64, 64),
    ]

    best_tile = benchmark_tile_sizes(256, 2048, 5632, tile_configs)  # MLP
    benchmark_tile_sizes(256, 2048, 2048, tile_configs)              # Attention

    # ── Part 2: num_warps and num_stages tuning ───────────────────────
    print("\n" + "=" * 50)
    print("PART 2: num_warps and num_stages Tuning")
    print("=" * 50)

    warp_stage_configs = [
        (1, 1),
        (2, 1),
        (4, 1),
        (8, 1),
        (4, 2),
        (4, 3),
        (8, 2),
        (8, 3),
    ]

    # Use best tile config from part 1 for MLP size
    TILE_M, TILE_N, TILE_K = best_tile
    benchmark_warps_stages(256, 2048, 5632, TILE_M, TILE_N, TILE_K, warp_stage_configs)
    benchmark_warps_stages(256, 2048, 2048, TILE_M, TILE_N, TILE_K, warp_stage_configs)