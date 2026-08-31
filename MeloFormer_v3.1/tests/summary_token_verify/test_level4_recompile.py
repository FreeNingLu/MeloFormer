#!/usr/bin/env python3
"""
Level 4: Captured Tensor 重编译测试

核心问题：当 captured tensor 的值改变（但 shape 不变）时，
torch.compile(flex_attention) 是否触发 Triton 重编译？

这是 v2.1 方案的关键分叉点：
- 不重编译 → 可以恢复 v1.7 的 captured tensor mask_mod（细粒度稀疏）
- 重编译 → 必须保持 v2.0 的纯算术 mask_mod

测试策略：
1. 用 torch._dynamo 的编译计数器检测重编译
2. 模拟真实场景：不同 batch 有不同的 chord_ids / token_type_ids
3. 测试多种 captured tensor 模式
"""

import torch
import torch.nn.functional as F
import time
import warnings
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

warnings.filterwarnings("ignore")

device = torch.device("cuda")
dtype = torch.bfloat16

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()


def count_compilations(fn, *args, **kwargs):
    """运行 fn 并返回 (结果, 是否触发了新编译)"""
    import torch._dynamo as dynamo
    before = dynamo.utils.CompileProfiler.totals if hasattr(dynamo.utils, 'CompileProfiler') else None

    # 用 _dynamo 的 frame count 检测
    frame_count_before = torch._dynamo.utils.counters["frames"]["ok"]
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    frame_count_after = torch._dynamo.utils.counters["frames"]["ok"]

    new_compilations = frame_count_after - frame_count_before
    return result, new_compilations


# ============================================================
# Test 4a: 1D captured tensor — 不同值，相同 shape
# ============================================================
print("=" * 60)
print("Test 4a: 1D captured tensor (chord_ids) — 值变化是否重编译")
print("=" * 60)

SEQ_LEN = 2048
NUM_HEADS = 8
HEAD_DIM = 64
CHORD_LEN = 128

# 编译 flex_attention
compiled_flex = torch.compile(flex_attention)

# --- 第一组 chord_ids ---
chord_ids_v1 = torch.arange(SEQ_LEN, device=device) // CHORD_LEN  # [0,0,...,1,1,...,15,15,...]

def make_mask_mod_1d(chord_ids):
    """创建使用 captured tensor 的 mask_mod"""
    def mask_mod(b, h, q_idx, kv_idx):
        same_chord = chord_ids[q_idx] == chord_ids[kv_idx]
        causal = q_idx >= kv_idx
        return same_chord & causal
    return mask_mod

mask_mod_v1 = make_mask_mod_1d(chord_ids_v1)
block_mask_v1 = create_block_mask(
    mask_mod_v1, B=1, H=1, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN, device=device
)

q = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
k = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
v = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)

# 第一次调用（预期触发编译）
torch._dynamo.utils.counters.clear()
print("  Call 1 (initial compile)...")
t0 = time.time()
out1, comp1 = count_compilations(compiled_flex, q, k, v, block_mask=block_mask_v1)
loss1 = out1.sum()
loss1.backward()
t1 = time.time()
print(f"    Time: {t1-t0:.2f}s, New compilations: {comp1}")

# --- 第二组 chord_ids（不同值，相同 shape）---
# 模拟不同 batch：chord 长度不均匀
chord_ids_v2 = torch.zeros(SEQ_LEN, dtype=torch.long, device=device)
# 不均匀 chord: chord0=200 tokens, chord1=100 tokens, chord2=150 tokens, ...
offsets = [0, 200, 300, 450, 600, 800, 950, 1100, 1250, 1400, 1550, 1700, 1850, 2048]
for i in range(len(offsets) - 1):
    chord_ids_v2[offsets[i]:offsets[i+1]] = i

mask_mod_v2 = make_mask_mod_1d(chord_ids_v2)
block_mask_v2 = create_block_mask(
    mask_mod_v2, B=1, H=1, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN, device=device
)

q.grad = None
k.grad = None
v.grad = None

# 第二次调用（关键：是否重编译？）
torch._dynamo.utils.counters.clear()
print("  Call 2 (different chord_ids values, same shape)...")
t0 = time.time()
out2, comp2 = count_compilations(compiled_flex, q, k, v, block_mask=block_mask_v2)
loss2 = out2.sum()
loss2.backward()
t1 = time.time()
print(f"    Time: {t1-t0:.2f}s, New compilations: {comp2}")

# --- 第三组（再换一次，确认稳定）---
chord_ids_v3 = torch.arange(SEQ_LEN, device=device) // 64  # 不同的 chord 长度
mask_mod_v3 = make_mask_mod_1d(chord_ids_v3)
block_mask_v3 = create_block_mask(
    mask_mod_v3, B=1, H=1, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN, device=device
)

q.grad = None
k.grad = None
v.grad = None

torch._dynamo.utils.counters.clear()
print("  Call 3 (yet another chord_ids, same shape)...")
t0 = time.time()
out3, comp3 = count_compilations(compiled_flex, q, k, v, block_mask=block_mask_v3)
loss3 = out3.sum()
loss3.backward()
t1 = time.time()
print(f"    Time: {t1-t0:.2f}s, New compilations: {comp3}")

test4a_pass = (comp2 == 0 and comp3 == 0)
print(f"  Recompilations after initial: call2={comp2}, call3={comp3}")
print(f"  {'PASSED — no recompilation!' if test4a_pass else 'FAILED — recompilation detected!'}")
print()


# ============================================================
# Test 4b: 2D captured tensor — type_visibility 查找表
# ============================================================
print("=" * 60)
print("Test 4b: 2D captured tensor (type_visibility) — 值变化")
print("=" * 60)

type_vis_v1 = torch.tensor([
    [1, 1, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
], dtype=torch.bool, device=device)

token_types_v1 = torch.randint(0, 4, (SEQ_LEN,), device=device)

def make_mask_mod_2d(chord_ids, token_types, type_vis):
    def mask_mod(b, h, q_idx, kv_idx):
        same_chord = chord_ids[q_idx] == chord_ids[kv_idx]
        causal = q_idx >= kv_idx
        q_type = token_types[q_idx]
        k_type = token_types[kv_idx]
        type_ok = type_vis[q_type, k_type]
        return same_chord & causal & type_ok
    return mask_mod

mask_mod_2d_v1 = make_mask_mod_2d(chord_ids_v1, token_types_v1, type_vis_v1)
block_mask_2d_v1 = create_block_mask(
    mask_mod_2d_v1, B=1, H=1, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN, device=device
)

compiled_flex_2 = torch.compile(flex_attention)

q2 = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
k2 = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
v2 = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)

# 初始编译
torch._dynamo.utils.counters.clear()
print("  Call 1 (initial compile)...")
t0 = time.time()
out_2d_1, comp_2d_1 = count_compilations(compiled_flex_2, q2, k2, v2, block_mask=block_mask_2d_v1)
out_2d_1.sum().backward()
t1 = time.time()
print(f"    Time: {t1-t0:.2f}s, New compilations: {comp_2d_1}")

# 换 token_types 和 chord_ids
token_types_v2 = torch.randint(0, 4, (SEQ_LEN,), device=device)
mask_mod_2d_v2 = make_mask_mod_2d(chord_ids_v2, token_types_v2, type_vis_v1)
block_mask_2d_v2 = create_block_mask(
    mask_mod_2d_v2, B=1, H=1, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN, device=device
)

q2.grad = None; k2.grad = None; v2.grad = None
torch._dynamo.utils.counters.clear()
print("  Call 2 (different chord_ids + token_types)...")
t0 = time.time()
out_2d_2, comp_2d_2 = count_compilations(compiled_flex_2, q2, k2, v2, block_mask=block_mask_2d_v2)
out_2d_2.sum().backward()
t1 = time.time()
print(f"    Time: {t1-t0:.2f}s, New compilations: {comp_2d_2}")

# 再换一次
token_types_v3 = torch.randint(0, 4, (SEQ_LEN,), device=device)
mask_mod_2d_v3 = make_mask_mod_2d(chord_ids_v3, token_types_v3, type_vis_v1)
block_mask_2d_v3 = create_block_mask(
    mask_mod_2d_v3, B=1, H=1, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN, device=device
)

q2.grad = None; k2.grad = None; v2.grad = None
torch._dynamo.utils.counters.clear()
print("  Call 3 (yet another set)...")
t0 = time.time()
out_2d_3, comp_2d_3 = count_compilations(compiled_flex_2, q2, k2, v2, block_mask=block_mask_2d_v3)
out_2d_3.sum().backward()
t1 = time.time()
print(f"    Time: {t1-t0:.2f}s, New compilations: {comp_2d_3}")

test4b_pass = (comp_2d_2 == 0 and comp_2d_3 == 0)
print(f"  Recompilations after initial: call2={comp_2d_2}, call3={comp_2d_3}")
print(f"  {'PASSED — no recompilation!' if test4b_pass else 'FAILED — recompilation detected!'}")
print()


# ============================================================
# Test 4c: 时间对比 — 编译 vs 复用
# ============================================================
print("=" * 60)
print("Test 4c: 时间稳定性 — 连续 10 次不同 tensor 值")
print("=" * 60)

compiled_flex_3 = torch.compile(flex_attention)
q3 = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
k3 = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
v3 = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)

times = []
compilations = []

for i in range(10):
    # 每次生成不同的 chord_ids
    if i == 0:
        chord_ids_i = torch.arange(SEQ_LEN, device=device) // CHORD_LEN
    else:
        # 随机 chord 长度
        lengths = torch.randint(50, 200, (30,))
        chord_ids_i = torch.zeros(SEQ_LEN, dtype=torch.long, device=device)
        pos = 0
        chord_idx = 0
        for length in lengths:
            end = min(pos + length.item(), SEQ_LEN)
            chord_ids_i[pos:end] = chord_idx
            chord_idx += 1
            pos = end
            if pos >= SEQ_LEN:
                break

    mask_mod_i = make_mask_mod_1d(chord_ids_i)
    block_mask_i = create_block_mask(
        mask_mod_i, B=1, H=1, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN, device=device
    )

    q3.grad = None; k3.grad = None; v3.grad = None
    torch._dynamo.utils.counters.clear()

    torch.cuda.synchronize()
    t0 = time.time()
    out_i, comp_i = count_compilations(compiled_flex_3, q3, k3, v3, block_mask=block_mask_i)
    out_i.sum().backward()
    torch.cuda.synchronize()
    t1 = time.time()

    times.append(t1 - t0)
    compilations.append(comp_i)
    print(f"  Iter {i}: {t1-t0:.3f}s, compilations: {comp_i}")

print()
print(f"  First call (compile): {times[0]:.3f}s")
print(f"  Subsequent avg: {sum(times[1:]) / len(times[1:]):.3f}s")
print(f"  Total recompilations after first: {sum(compilations[1:])}")

test4c_pass = sum(compilations[1:]) == 0
print(f"  {'PASSED — stable execution!' if test4c_pass else 'FAILED — recompilations detected!'}")
print()


# ============================================================
# Test 4d: block_mask 重建 vs flex_attention 重编译 区分
# ============================================================
print("=" * 60)
print("Test 4d: 区分 block_mask 重建成本 vs kernel 重编译成本")
print("=" * 60)

# 预先创建 5 个不同的 block_mask
masks = []
for i in range(5):
    chord_ids_i = torch.arange(SEQ_LEN, device=device) // (CHORD_LEN + i * 10)
    mask_mod_i = make_mask_mod_1d(chord_ids_i)
    masks.append(create_block_mask(
        mask_mod_i, B=1, H=1, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN, device=device
    ))

compiled_flex_4 = torch.compile(flex_attention)
q4 = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
k4 = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
v4 = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)

# 初始编译
torch._dynamo.utils.counters.clear()
out_init = compiled_flex_4(q4, k4, v4, block_mask=masks[0])
out_init.sum().backward()
q4.grad = None; k4.grad = None; v4.grad = None

# 用预建的 mask 轮流调用
print("  Cycling through 5 pre-built block_masks (3 rounds):")
total_recomp = 0
for round_idx in range(3):
    for mask_idx in range(5):
        q4.grad = None; k4.grad = None; v4.grad = None
        torch._dynamo.utils.counters.clear()
        torch.cuda.synchronize()
        t0 = time.time()
        out_d, comp_d = count_compilations(compiled_flex_4, q4, k4, v4, block_mask=masks[mask_idx])
        out_d.sum().backward()
        torch.cuda.synchronize()
        t1 = time.time()
        total_recomp += comp_d
        if comp_d > 0:
            print(f"    Round {round_idx}, mask {mask_idx}: RECOMPILED! ({t1-t0:.3f}s)")

print(f"  Total recompilations across 15 calls: {total_recomp}")
test4d_pass = total_recomp == 0
print(f"  {'PASSED — zero recompilations!' if test4d_pass else 'FAILED — recompilations detected!'}")
print()


# ============================================================
# Summary
# ============================================================
print("=" * 60)
print("Level 4 Summary")
print("=" * 60)
print(f"  4a (1D tensor, value change):     {'PASS' if test4a_pass else 'FAIL'}")
print(f"  4b (2D tensor, value change):     {'PASS' if test4b_pass else 'FAIL'}")
print(f"  4c (10x stability):               {'PASS' if test4c_pass else 'FAIL'}")
print(f"  4d (pre-built mask cycling):      {'PASS' if test4d_pass else 'FAIL'}")
print()

all_pass = test4a_pass and test4b_pass and test4c_pass and test4d_pass
if all_pass:
    print("ALL PASSED — captured tensor 不触发重编译！")
    print("→ 可以恢复 v1.7 的 captured tensor mask_mod")
    print("→ v2.1 完整方案可行")
else:
    print("SOME FAILED — captured tensor 触发重编译")
    print("→ 必须保持 v2.0 的纯算术 mask_mod")
    print("→ v2.1 精简方案")
    if not test4a_pass:
        print("  ! 1D tensor indexing 触发重编译")
    if not test4b_pass:
        print("  ! 2D tensor indexing 触发重编译")
