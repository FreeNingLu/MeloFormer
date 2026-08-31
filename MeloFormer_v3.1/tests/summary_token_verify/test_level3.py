#!/usr/bin/env python3
"""
Level 3 PoC: 两阶段完整 pipeline
Phase 1 (SDPA) -> Linear -> Phase 2 (FlexAttention) + GQA + BF16 + GC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.utils.checkpoint import checkpoint

torch.manual_seed(42)
device = "cuda"
dtype = torch.bfloat16

compiled_flex_attention = torch.compile(flex_attention)
compiled_create_block_mask = torch.compile(create_block_mask)

# ============================================================
# 配置 (MeloFormer Large)
# ============================================================
D_MODEL = 768
N_HEADS = 12
N_KV_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 64
N_REP = N_HEADS // N_KV_HEADS  # 3
KV_DIM = HEAD_DIM * N_KV_HEADS  # 256

SEQ_LEN = 4096
NUM_CHORDS = 32
TOTAL_KV = NUM_CHORDS + SEQ_LEN
BATCH_SIZE = 1

# ============================================================
# Metadata tensors
# ============================================================
tokens_per_chord = SEQ_LEN // NUM_CHORDS
chord_ids = (torch.arange(SEQ_LEN, device=device) // tokens_per_chord).clamp(0, NUM_CHORDS - 1)
chord_ids_seg = chord_ids.clone()
instrument_ids = torch.arange(SEQ_LEN, device=device) % 8
token_type_ids = torch.arange(SEQ_LEN, device=device) % 4
note_ids = torch.arange(SEQ_LEN, device=device) // 4
type_visibility = torch.tensor([
    [True, True, False, False],
    [True, True, False, False],
    [False, False, False, False],
    [False, False, False, True],
], dtype=torch.bool, device=device)

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Config: D={D_MODEL}, H={N_HEADS}, KV_H={N_KV_HEADS}, SEQ={SEQ_LEN}, CHORDS={NUM_CHORDS}")
print()

# ============================================================
# Phase 1 的显式 mask (SDPA 用)
# ============================================================
print("Building Phase 1 explicit mask...")
with torch.no_grad():
    ss_mask = torch.tril(torch.ones(NUM_CHORDS, NUM_CHORDS, dtype=torch.bool, device=device))
    sr_mask = torch.zeros(NUM_CHORDS, SEQ_LEN, dtype=torch.bool, device=device)
    for i in range(NUM_CHORDS):
        sr_mask[i] = (chord_ids == i)
    phase1_mask = torch.cat([ss_mask, sr_mask], dim=1)
    phase1_mask = phase1_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, M, M+N)
print(f"  Phase 1 mask shape: {phase1_mask.shape}")

# ============================================================
# Phase 2 的 updating_mask_mod (FlexAttention 用)
# ============================================================
def updating_mask_mod(b, h, q_idx, kv_idx):
    q_safe = q_idx.clamp(0, SEQ_LEN - 1)
    kv_safe = kv_idx.clamp(0, TOTAL_KV - 1)
    is_kv_summary = kv_safe < NUM_CHORDS
    is_kv_regular = ~is_kv_summary
    q_chord = chord_ids[q_safe].clamp(0, NUM_CHORDS - 1)

    # RS
    rs_mask = is_kv_summary & (q_chord > kv_safe)

    # RR
    reg_kv = (kv_safe - NUM_CHORDS).clamp(0, SEQ_LEN - 1)
    rr_causal = q_safe >= reg_kv
    q_chord_seg = chord_ids_seg[q_safe]
    k_chord_seg = chord_ids_seg[reg_kv]
    q_inst = instrument_ids[q_safe]
    k_inst = instrument_ids[reg_kv]
    is_global_k = (k_chord_seg == -1) | (k_inst == 129)
    same_inst = (q_inst == k_inst) & (q_inst < 129) & (k_inst < 129)
    diff_inst = (q_inst != k_inst) & (q_inst < 129) & (k_inst < 129)
    chord_diff = q_chord_seg - k_chord_seg
    cross_near = diff_inst & (chord_diff >= 0) & (chord_diff <= 2)
    cross_far = diff_inst & (chord_diff == 4)
    chord_mask = is_global_k | same_inst | cross_near | cross_far

    q_type = token_type_ids[q_safe].clamp(0, 3)
    k_type = token_type_ids[reg_kv].clamp(0, 3)
    q_note = note_ids[q_safe]
    k_note = note_ids[reg_kv]
    same_note = (q_note == k_note) & (q_note >= 0) & (k_note >= 0)
    type_visible = type_visibility[q_type, k_type]
    is_global_type = (token_type_ids[q_safe] == -1) | (token_type_ids[reg_kv] == -1)
    is_global_note = (q_note == -1) | (k_note == -1)
    token_mask = same_note | type_visible | is_global_type | is_global_note

    rr_mask = is_kv_regular & rr_causal & chord_mask & token_mask
    return rs_mask | rr_mask

print("Creating Phase 2 block mask...")
phase2_block_mask = compiled_create_block_mask(
    updating_mask_mod, B=BATCH_SIZE, H=None,
    Q_LEN=SEQ_LEN, KV_LEN=TOTAL_KV, device=device,
)
print(f"  Phase 2 sparsity: {phase2_block_mask.sparsity():.1f}%")

# ============================================================
# MinimalSummaryAttention 模块
# ============================================================
class MinimalSummaryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Phase 1 projections
        self.sum_q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.sum_k_proj = nn.Linear(D_MODEL, KV_DIM)
        self.sum_v_proj = nn.Linear(D_MODEL, KV_DIM)
        self.reg_q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.reg_k_proj = nn.Linear(D_MODEL, KV_DIM)
        self.reg_v_proj = nn.Linear(D_MODEL, KV_DIM)
        # Phase 1->2 bridge
        self.sum_k2_proj = nn.Linear(D_MODEL, KV_DIM)
        self.sum_v2_proj = nn.Linear(D_MODEL, KV_DIM)
        # Output
        self.sum_out_proj = nn.Linear(D_MODEL, D_MODEL)
        self.reg_out_proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, sum_x, reg_x, phase1_mask, phase2_block_mask):
        B, M, D = sum_x.shape
        N = reg_x.shape[1]

        # === Projections ===
        sum_q = self.sum_q_proj(sum_x).view(B, M, N_HEADS, HEAD_DIM).transpose(1, 2)
        sum_k = self.sum_k_proj(sum_x).view(B, M, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        sum_v = self.sum_v_proj(sum_x).view(B, M, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        reg_q = self.reg_q_proj(reg_x).view(B, N, N_HEADS, HEAD_DIM).transpose(1, 2)
        reg_k = self.reg_k_proj(reg_x).view(B, N, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        reg_v = self.reg_v_proj(reg_x).view(B, N, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

        # === GQA: repeat_interleave ===
        sum_k = sum_k.repeat_interleave(N_REP, dim=1)
        sum_v = sum_v.repeat_interleave(N_REP, dim=1)
        reg_k = reg_k.repeat_interleave(N_REP, dim=1)
        reg_v = reg_v.repeat_interleave(N_REP, dim=1)

        # === Phase 1: Summarize (SDPA) ===
        cat_k1 = torch.cat([sum_k, reg_k], dim=2)
        cat_v1 = torch.cat([sum_v, reg_v], dim=2)
        sum_out = F.scaled_dot_product_attention(
            sum_q, cat_k1, cat_v1, attn_mask=phase1_mask
        )
        sum_out = sum_out.transpose(1, 2).reshape(B, M, D)

        # === Phase 1->2: K2/V2 projection ===
        k2 = self.sum_k2_proj(sum_out).view(B, M, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v2 = self.sum_v2_proj(sum_out).view(B, M, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        k2 = k2.repeat_interleave(N_REP, dim=1)
        v2 = v2.repeat_interleave(N_REP, dim=1)

        # === Phase 2: Updating (compiled FlexAttention) ===
        cat_k2 = torch.cat([k2, reg_k], dim=2)
        cat_v2 = torch.cat([v2, reg_v], dim=2)
        reg_out = compiled_flex_attention(reg_q, cat_k2, cat_v2, block_mask=phase2_block_mask)
        reg_out = reg_out.transpose(1, 2).reshape(B, N, D)

        # === Output projections ===
        return self.sum_out_proj(sum_out), self.reg_out_proj(reg_out)

# ============================================================
# Test 3a: Forward + Backward (BF16)
# ============================================================
print()
print("=" * 60)
print("Test 3a: Forward + Backward (BF16)")
print("=" * 60)

model = MinimalSummaryAttention().to(device=device, dtype=dtype)
sum_x = torch.randn(BATCH_SIZE, NUM_CHORDS, D_MODEL, device=device, dtype=dtype, requires_grad=True)
reg_x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=device, dtype=dtype, requires_grad=True)

sum_out, reg_out = model(sum_x, reg_x, phase1_mask, phase2_block_mask)
loss = sum_out.sum() + reg_out.sum()
loss.backward()

assert sum_x.grad is not None and sum_x.grad.abs().sum() > 0
assert reg_x.grad is not None and reg_x.grad.abs().sum() > 0
print(f"  sum_out shape: {sum_out.shape}")
print(f"  reg_out shape: {reg_out.shape}")
print(f"  sum_x.grad norm: {sum_x.grad.norm().item():.6f}")
print(f"  reg_x.grad norm: {reg_x.grad.norm().item():.6f}")
print(f"  All param grads exist: {all(p.grad is not None for p in model.parameters())}")
print("  PASSED")

# ============================================================
# Test 3b: Gradient Checkpointing
# ============================================================
print()
print("=" * 60)
print("Test 3b: Gradient Checkpointing")
print("=" * 60)

model.zero_grad()
sum_x2 = sum_x.detach().clone().requires_grad_(True)
reg_x2 = reg_x.detach().clone().requires_grad_(True)

sum_out2, reg_out2 = checkpoint(
    model, sum_x2, reg_x2, phase1_mask, phase2_block_mask,
    use_reentrant=False,
)
loss2 = sum_out2.sum() + reg_out2.sum()
loss2.backward()

assert sum_x2.grad is not None and sum_x2.grad.abs().sum() > 0
assert reg_x2.grad is not None and reg_x2.grad.abs().sum() > 0
print(f"  sum_x.grad norm: {sum_x2.grad.norm().item():.6f}")
print(f"  reg_x.grad norm: {reg_x2.grad.norm().item():.6f}")
print("  PASSED")

# ============================================================
# Test 3c: torch.compile 整个模块
# ============================================================
print()
print("=" * 60)
print("Test 3c: torch.compile full module")
print("=" * 60)

compiled_model = torch.compile(model, mode="default")
model.zero_grad()
sum_x3 = sum_x.detach().clone().requires_grad_(True)
reg_x3 = reg_x.detach().clone().requires_grad_(True)

sum_out3, reg_out3 = compiled_model(sum_x3, reg_x3, phase1_mask, phase2_block_mask)
loss3 = sum_out3.sum() + reg_out3.sum()
loss3.backward()

assert sum_x3.grad is not None and sum_x3.grad.abs().sum() > 0
print(f"  Compiled forward+backward: OK")
print(f"  sum_x.grad norm: {sum_x3.grad.norm().item():.6f}")
print(f"  reg_x.grad norm: {reg_x3.grad.norm().item():.6f}")
print("  PASSED")

# ============================================================
# Test 3d: 显存峰值测量
# ============================================================
print()
print("=" * 60)
print("Test 3d: Peak memory measurement")
print("=" * 60)

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

model.zero_grad()
sum_x4 = torch.randn(BATCH_SIZE, NUM_CHORDS, D_MODEL, device=device, dtype=dtype, requires_grad=True)
reg_x4 = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=device, dtype=dtype, requires_grad=True)

sum_out4, reg_out4 = model(sum_x4, reg_x4, phase1_mask, phase2_block_mask)
loss4 = sum_out4.sum() + reg_out4.sum()
loss4.backward()

peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
print(f"  Peak GPU memory: {peak_mem:.2f} GB")
print(f"  Config: B={BATCH_SIZE}, N={SEQ_LEN}, CHORDS={NUM_CHORDS}, D={D_MODEL}, H={N_HEADS}")
print("  PASSED")

print()
print("=" * 60)
print("Level 3 ALL PASSED")
print("=" * 60)
