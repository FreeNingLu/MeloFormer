#!/usr/bin/env python3
"""
Level 1 PoC: FlexAttention 基础操作验证
验证 captured tensor indexing, 2D indexing, clamp + boolean combo
使用 torch.compile 确保走 Triton fused kernel（非 dense fallback）
"""
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

torch.manual_seed(42)
device = "cuda"
dtype = torch.bfloat16
SEQ_LEN = 2048

# 关键：compile flex_attention 使其走 Triton fused kernel
compiled_flex_attention = torch.compile(flex_attention)

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"SM: {torch.cuda.get_device_capability(0)}")
print(f"dtype: {dtype}, SEQ_LEN: {SEQ_LEN}")
print()

# ============================================================
# Test 1a: 1D captured tensor indexing (chord-causal mask)
# ============================================================
print("=" * 60)
print("Test 1a: 1D captured tensor indexing (chord-causal)")
print("=" * 60)

chord_ids = torch.randint(0, 8, (SEQ_LEN,), device=device)

def mask_mod_1d(b, h, q_idx, kv_idx):
    """最小的 captured tensor indexing: chord-causal mask"""
    kv_safe = kv_idx.clamp(0, SEQ_LEN - 1)
    q_safe = q_idx.clamp(0, SEQ_LEN - 1)
    q_chord = chord_ids[q_safe]
    k_chord = chord_ids[kv_safe]
    return q_chord >= k_chord

block_mask_1d = create_block_mask(
    mask_mod_1d, B=1, H=1, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN, device=device
)

q = torch.randn(1, 8, SEQ_LEN, 64, device=device, dtype=dtype, requires_grad=True)
k = torch.randn(1, 8, SEQ_LEN, 64, device=device, dtype=dtype, requires_grad=True)
v = torch.randn(1, 8, SEQ_LEN, 64, device=device, dtype=dtype, requires_grad=True)

# Forward
out = compiled_flex_attention(q, k, v, block_mask=block_mask_1d)
# Backward
loss = out.sum()
loss.backward()

assert q.grad is not None and q.grad.abs().sum() > 0, "q.grad is zero!"
assert k.grad is not None and k.grad.abs().sum() > 0, "k.grad is zero!"
assert v.grad is not None and v.grad.abs().sum() > 0, "v.grad is zero!"
print(f"  Forward output shape: {out.shape}")
print(f"  q.grad norm: {q.grad.norm().item():.4f}")
print(f"  k.grad norm: {k.grad.norm().item():.4f}")
print(f"  v.grad norm: {v.grad.norm().item():.4f}")
print("  PASSED")

# ============================================================
# Test 1b: 2D captured tensor indexing (type visibility)
# ============================================================
print()
print("=" * 60)
print("Test 1b: 2D captured tensor indexing")
print("=" * 60)

type_visibility = torch.tensor([
    [True,  True,  False, False],
    [True,  True,  False, False],
    [False, False, False, False],
    [False, False, False, True ],
], dtype=torch.bool, device=device)

token_types = torch.randint(0, 4, (SEQ_LEN,), device=device)

def mask_mod_2d(b, h, q_idx, kv_idx):
    """2D captured tensor indexing: type visibility lookup"""
    q_safe = q_idx.clamp(0, SEQ_LEN - 1)
    kv_safe = kv_idx.clamp(0, SEQ_LEN - 1)
    q_type = token_types[q_safe]
    k_type = token_types[kv_safe]
    causal = q_idx >= kv_idx
    return causal & type_visibility[q_type, k_type]

block_mask_2d = create_block_mask(
    mask_mod_2d, B=1, H=1, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN, device=device
)

q2 = torch.randn(1, 8, SEQ_LEN, 64, device=device, dtype=dtype, requires_grad=True)
k2 = torch.randn(1, 8, SEQ_LEN, 64, device=device, dtype=dtype, requires_grad=True)
v2 = torch.randn(1, 8, SEQ_LEN, 64, device=device, dtype=dtype, requires_grad=True)

out2 = compiled_flex_attention(q2, k2, v2, block_mask=block_mask_2d)
out2.sum().backward()

assert q2.grad is not None and q2.grad.abs().sum() > 0, "q2.grad is zero!"
print(f"  Forward output shape: {out2.shape}")
print(f"  q.grad norm: {q2.grad.norm().item():.4f}")
print("  PASSED")

# ============================================================
# Test 1c: clamp + boolean combo (updating_mask_mod pattern)
# ============================================================
print()
print("=" * 60)
print("Test 1c: clamp + boolean combo (updating_mask_mod pattern)")
print("=" * 60)

NUM_CHORDS = 32
chord_ids_full = torch.randint(0, NUM_CHORDS, (SEQ_LEN,), device=device)
inst_ids = torch.randint(0, 8, (SEQ_LEN,), device=device)
chord_ids_seg = torch.randint(0, NUM_CHORDS, (SEQ_LEN,), device=device)

TOTAL_KV = NUM_CHORDS + SEQ_LEN

def mask_mod_complex(b, h, q_idx, kv_idx):
    """模拟 updating_mask_mod 的核心操作模式"""
    q_safe = q_idx.clamp(0, SEQ_LEN - 1)
    kv_safe = kv_idx.clamp(0, TOTAL_KV - 1)

    is_kv_summary = kv_safe < NUM_CHORDS
    is_kv_regular = ~is_kv_summary

    q_chord = chord_ids_full[q_safe].clamp(0, NUM_CHORDS - 1)

    # RS: Regular -> Summary (causal on chord)
    rs_mask = is_kv_summary & (q_chord > kv_safe)

    # RR: Regular -> Regular
    reg_kv = (kv_safe - NUM_CHORDS).clamp(0, SEQ_LEN - 1)
    rr_causal = q_safe >= reg_kv

    q_inst = inst_ids[q_safe]
    k_inst = inst_ids[reg_kv]
    q_chord_seg = chord_ids_seg[q_safe]
    k_chord_seg = chord_ids_seg[reg_kv]

    same_inst = (q_inst == k_inst) & (q_inst < 129) & (k_inst < 129)
    diff_inst = (q_inst != k_inst) & (q_inst < 129) & (k_inst < 129)
    chord_diff = q_chord_seg - k_chord_seg
    cross_near = diff_inst & (chord_diff >= 0) & (chord_diff <= 2)
    is_global = (k_chord_seg == -1) | (k_inst == 129)

    chord_mask = is_global | same_inst | cross_near
    rr_mask = is_kv_regular & rr_causal & chord_mask

    return rs_mask | rr_mask

block_mask_complex = create_block_mask(
    mask_mod_complex, B=1, H=None,
    Q_LEN=SEQ_LEN, KV_LEN=TOTAL_KV, device=device
)

q3 = torch.randn(1, 8, SEQ_LEN, 64, device=device, dtype=dtype, requires_grad=True)
k3 = torch.randn(1, 8, TOTAL_KV, 64, device=device, dtype=dtype, requires_grad=True)
v3 = torch.randn(1, 8, TOTAL_KV, 64, device=device, dtype=dtype, requires_grad=True)

out3 = compiled_flex_attention(q3, k3, v3, block_mask=block_mask_complex)
out3.sum().backward()

assert q3.grad is not None and q3.grad.abs().sum() > 0, "q3.grad is zero!"
print(f"  Forward output shape: {out3.shape}")
print(f"  Q_LEN={SEQ_LEN}, KV_LEN={TOTAL_KV} (asymmetric)")
print(f"  q.grad norm: {q3.grad.norm().item():.4f}")
print("  PASSED")

print()
print("=" * 60)
print("Level 1 ALL PASSED")
print("=" * 60)
