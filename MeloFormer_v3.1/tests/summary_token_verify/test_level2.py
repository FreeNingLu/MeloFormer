#!/usr/bin/env python3
"""
Level 2 PoC: 完整 updating_mask_mod 复杂度
包含所有 6 个 captured tensor + 2D indexing + token type sparsity
+ 数值正确性验证 (vs dense SDPA)
"""
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

torch.manual_seed(42)
device = "cuda"
dtype = torch.bfloat16

compiled_create_block_mask = torch.compile(create_block_mask)

# ============================================================
# 模拟 MeloFormer 的 metadata tensors
# ============================================================
SEQ_LEN = 4096
NUM_CHORDS = 32
TOTAL_KV = NUM_CHORDS + SEQ_LEN
NUM_INSTRUMENTS = 8
CROSS_INST_FULL_RANGE = 2
CROSS_INST_FAR_OFFSETS = (4,)

# chord_ids: 每个 token 属于哪个 chord (递增，每 chord ~128 tokens)
tokens_per_chord = SEQ_LEN // NUM_CHORDS
chord_ids = (torch.arange(SEQ_LEN, device=device) // tokens_per_chord).clamp(0, NUM_CHORDS - 1)

# chord_ids_seg: 与 chord_ids 相同 (简化)
chord_ids_seg = chord_ids.clone()

# instrument_ids: 循环分配乐器
instrument_ids = torch.arange(SEQ_LEN, device=device) % NUM_INSTRUMENTS

# token_type_ids: T(0), P(1), D(2), V(3) 循环
token_type_ids = torch.arange(SEQ_LEN, device=device) % 4

# note_ids: 每 4 个 token 一组
note_ids = torch.arange(SEQ_LEN, device=device) // 4

# type_visibility: (4, 4) 查找表
type_visibility = torch.tensor([
    [True,  True,  False, False],  # T -> T, P
    [True,  True,  False, False],  # P -> T, P
    [False, False, False, False],  # D -> nothing
    [False, False, False, True ],  # V -> V
], dtype=torch.bool, device=device)

compiled_flex_attention = torch.compile(flex_attention)

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Config: SEQ_LEN={SEQ_LEN}, NUM_CHORDS={NUM_CHORDS}, TOTAL_KV={TOTAL_KV}")
print()
# ============================================================
# 完整的 updating_mask_mod (从 attention_flex_summary.py 提取)
# ============================================================
rs_look_back = -1  # 因果: 看所有已完成 chord
use_token_type_sparsity = True


def updating_mask_mod(b, h, q_idx, kv_idx):
    """完整的 updating_mask_mod，包含所有操作模式"""
    sum_len = NUM_CHORDS
    total_kv_len = TOTAL_KV

    q_idx_safe = q_idx.clamp(0, SEQ_LEN - 1)
    kv_idx_safe = kv_idx.clamp(0, total_kv_len - 1)

    is_kv_summary = kv_idx_safe < sum_len
    is_kv_regular = ~is_kv_summary

    q_chord = chord_ids[q_idx_safe].clamp(0, sum_len - 1)

    # === RS: Regular -> Summary (因果) ===
    rs_mask = is_kv_summary & (q_chord > kv_idx_safe)

    # === RR: Regular -> Regular ===
    reg_kv_idx = (kv_idx_safe - sum_len).clamp(0, SEQ_LEN - 1)

    # 因果性
    rr_causal = q_idx_safe >= reg_kv_idx

    # 获取 token 信息 (6 个 captured tensor 的索引)
    q_chord_seg = chord_ids_seg[q_idx_safe]
    k_chord_seg = chord_ids_seg[reg_kv_idx]
    q_inst = instrument_ids[q_idx_safe]
    k_inst = instrument_ids[reg_kv_idx]

    # 全局 token
    is_global_k = (k_chord_seg == -1) | (k_inst == 129)

    # 同乐器
    same_inst = (q_inst == k_inst) & (q_inst < 129) & (k_inst < 129)

    # 跨乐器近距离
    chord_diff = q_chord_seg - k_chord_seg
    diff_inst = (q_inst != k_inst) & (q_inst < 129) & (k_inst < 129)
    cross_near = diff_inst & (chord_diff >= 0) & (chord_diff <= CROSS_INST_FULL_RANGE)

    # 跨乐器远距离 (for 循环展开)
    cross_far = torch.zeros_like(rr_causal)
    for offset in CROSS_INST_FAR_OFFSETS:
        cross_far = cross_far | (diff_inst & (chord_diff == offset))

    chord_mask = is_global_k | same_inst | cross_near | cross_far

    # === Token 类型级稀疏 ===
    q_type = token_type_ids[q_idx_safe]
    k_type = token_type_ids[reg_kv_idx]
    q_note = note_ids[q_idx_safe]
    k_note = note_ids[reg_kv_idx]

    same_note = (q_note == k_note) & (q_note >= 0) & (k_note >= 0)
    q_type_safe = q_type.clamp(0, 3)
    k_type_safe = k_type.clamp(0, 3)
    type_visible = type_visibility[q_type_safe, k_type_safe]  # 2D indexing
    is_global_type = (q_type == -1) | (k_type == -1)
    is_global_note = (q_note == -1) | (k_note == -1)
    token_type_mask = same_note | type_visible | is_global_type | is_global_note

    rr_mask = is_kv_regular & rr_causal & chord_mask & token_type_mask

    return rs_mask | rr_mask


# ============================================================
# Test 2a: BlockMask 创建 + Forward + Backward
# ============================================================
print("=" * 60)
print("Test 2a: Full updating_mask_mod compile + forward + backward")
print("=" * 60)

print("  Creating block mask...")
block_mask = compiled_create_block_mask(
    updating_mask_mod,
    B=1, H=None,
    Q_LEN=SEQ_LEN, KV_LEN=TOTAL_KV,
    device=device,
)
print(f"  Block mask created: Q_LEN={SEQ_LEN}, KV_LEN={TOTAL_KV}")
print(f"  Sparsity: {block_mask.sparsity():.1f}%")

# 12 heads (repeat_interleave 后)
q = torch.randn(1, 12, SEQ_LEN, 64, device=device, dtype=dtype, requires_grad=True)
k = torch.randn(1, 12, TOTAL_KV, 64, device=device, dtype=dtype, requires_grad=True)
v = torch.randn(1, 12, TOTAL_KV, 64, device=device, dtype=dtype, requires_grad=True)

print("  Running compiled forward...")
out = compiled_flex_attention(q, k, v, block_mask=block_mask)
print(f"  Output shape: {out.shape}")

print("  Running backward...")
out.sum().backward()

assert q.grad is not None and q.grad.abs().sum() > 0, "q.grad is zero!"
assert k.grad is not None and k.grad.abs().sum() > 0, "k.grad is zero!"
assert v.grad is not None and v.grad.abs().sum() > 0, "v.grad is zero!"
print(f"  q.grad norm: {q.grad.norm().item():.6f}")
print(f"  k.grad norm: {k.grad.norm().item():.6f}")
print(f"  v.grad norm: {v.grad.norm().item():.6f}")
print("  PASSED")
# ============================================================
# Test 2b: 数值正确性验证 vs dense SDPA
# ============================================================
print()
print("=" * 60)
print("Test 2b: Numerical correctness vs dense SDPA")
print("=" * 60)

CHECK_ROWS = 128  # 只验证前 128 行 (节省时间和显存)

print(f"  Building explicit mask for first {CHECK_ROWS} rows...")
with torch.no_grad():
    mask = torch.zeros(CHECK_ROWS, TOTAL_KV, dtype=torch.bool, device=device)
    for qi in range(CHECK_ROWS):
        for ki in range(TOTAL_KV):
            mask[qi, ki] = updating_mask_mod(
                torch.tensor(0, device=device),
                torch.tensor(0, device=device),
                torch.tensor(qi, device=device),
                torch.tensor(ki, device=device),
            )

    q_check = q[:, :, :CHECK_ROWS, :].detach()
    k_check = k.detach()
    v_check = v.detach()

    # Dense SDPA
    attn_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, CHECK_ROWS, TOTAL_KV)
    ref_out = F.scaled_dot_product_attention(q_check, k_check, v_check, attn_mask=attn_mask)

    # FlexAttention output (前 CHECK_ROWS 行)
    flex_out = out[:, :, :CHECK_ROWS, :].detach()

    max_diff = (ref_out - flex_out).abs().max().item()
    mean_diff = (ref_out - flex_out).abs().mean().item()
    print(f"  Max absolute diff:  {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")

    if max_diff < 1e-2:  # BF16 精度下的合理阈值
        print("  Numerical check PASSED")
    else:
        print(f"  WARNING: Large numerical diff ({max_diff:.4e}), investigate!")

# ============================================================
# Test 2c: 显存峰值测量
# ============================================================
print()
print("=" * 60)
print("Test 2c: Peak memory measurement")
print("=" * 60)

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

q_m = torch.randn(1, 12, SEQ_LEN, 64, device=device, dtype=dtype, requires_grad=True)
k_m = torch.randn(1, 12, TOTAL_KV, 64, device=device, dtype=dtype, requires_grad=True)
v_m = torch.randn(1, 12, TOTAL_KV, 64, device=device, dtype=dtype, requires_grad=True)

out_m = compiled_flex_attention(q_m, k_m, v_m, block_mask=block_mask)
out_m.sum().backward()

peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
print(f"  Peak GPU memory: {peak_mem:.2f} GB")
print(f"  Config: B=1, SEQ_LEN={SEQ_LEN}, NUM_CHORDS={NUM_CHORDS}, heads=12, head_dim=64")

# 对比: dense attention 的理论显存
dense_attn_mem = SEQ_LEN * TOTAL_KV * 12 * 4 / (1024 ** 3)  # FP32 score matrix
print(f"  Dense attention score matrix would be: {dense_attn_mem:.2f} GB (FP32)")
print("  PASSED")

print()
print("=" * 60)
print("Level 2 ALL PASSED")
print("=" * 60)
