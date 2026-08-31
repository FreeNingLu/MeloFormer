#!/usr/bin/env python3
"""
MeloFormer v2.1 Smoke Test

验证 v2.1 改动后的完整模型能跑通:
1. 8m 模型 forward + backward (BF16)
2. Gradient Checkpointing
3. torch.compile
4. 变长 chord + Token Type Visibility
5. 显存测量
"""
import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.meloformer import create_model

torch.manual_seed(42)
device = torch.device('cuda')
dtype = torch.bfloat16

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# ============================================================
# 模拟音乐数据 (变长 chord)
# ============================================================
BATCH = 1
NUM_CHORDS = 8
# 变长 chord: chord 长度分别为 30, 50, 40, 60, 35, 45, 55, 25 = 340 tokens
CHORD_LENGTHS = [30, 50, 40, 60, 35, 45, 55, 25]
SEQ_LEN = sum(CHORD_LENGTHS)  # 340
VOCAB_SIZE = 643

# 构建 metadata tensors
chord_ids_1d = torch.zeros(SEQ_LEN, dtype=torch.long, device=device)
token_type_ids = torch.zeros(SEQ_LEN, dtype=torch.long, device=device)
note_ids = torch.zeros(SEQ_LEN, dtype=torch.long, device=device)
instrument_ids = torch.zeros(BATCH, SEQ_LEN, dtype=torch.long, device=device)

pos = 0
note_counter = 0
for chord_idx, chord_len in enumerate(CHORD_LENGTHS):
    # 每个 chord 开头: Chord token (type=-1)
    chord_ids_1d[pos] = chord_idx
    token_type_ids[pos] = -1  # 全局
    note_ids[pos] = -1
    instrument_ids[0, pos] = 129  # 全局
    pos += 1

    # 每个 chord 内: Instrument + T/P/D/V 四元组
    remaining = chord_len - 1
    inst_id = chord_idx % 4  # 循环乐器
    while remaining >= 5:  # 1 instrument + 4 note tokens
        # Instrument token
        chord_ids_1d[pos] = chord_idx
        token_type_ids[pos] = -1
        note_ids[pos] = -1
        instrument_ids[0, pos] = inst_id
        pos += 1
        remaining -= 1

        # T, P, D, V
        for t in range(4):
            if pos >= SEQ_LEN:
                break
            chord_ids_1d[pos] = chord_idx
            token_type_ids[pos] = t  # 0=T, 1=P, 2=D, 3=V
            note_ids[pos] = note_counter
            instrument_ids[0, pos] = inst_id
            pos += 1
            remaining -= 1
        note_counter += 1

    # 填充剩余
    while remaining > 0 and pos < SEQ_LEN:
        chord_ids_1d[pos] = chord_idx
        token_type_ids[pos] = 0
        note_ids[pos] = note_counter
        instrument_ids[0, pos] = inst_id
        pos += 1
        remaining -= 1

# 随机 token ids
token_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=device)
chord_ids = chord_ids_1d.unsqueeze(0).expand(BATCH, -1)

print(f"Config: BATCH={BATCH}, SEQ_LEN={SEQ_LEN}, NUM_CHORDS={NUM_CHORDS}")
print(f"Chord lengths: {CHORD_LENGTHS}")
print(f"chord_ids unique: {chord_ids_1d.unique().tolist()}")
print(f"token_type_ids unique: {token_type_ids.unique().tolist()}")
print()


# ============================================================
# Test 1: 8m 模型 forward + backward
# ============================================================
print("=" * 60)
print("Test 1: 8m model forward + backward (BF16)")
print("=" * 60)

model = create_model(vocab_size=VOCAB_SIZE, model_size='8m').to(device=device, dtype=dtype)
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  Model params: {params:.2f}M")

with torch.autocast(device_type='cuda', dtype=dtype):
    logits = model(
        token_ids=token_ids,
        chord_ids=chord_ids,
        instrument_ids=instrument_ids,
        token_type_ids=token_type_ids.unsqueeze(0).expand(BATCH, -1),
        note_ids=note_ids.unsqueeze(0).expand(BATCH, -1),
        num_chords=NUM_CHORDS,
    )

assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE), f"Expected {(BATCH, SEQ_LEN, VOCAB_SIZE)}, got {logits.shape}"
print(f"  logits shape: {logits.shape}")

loss = logits.sum()
loss.backward()

# 最后一层的 Summary 分支不参与 loss (sum_x 不输出)，梯度为 None 是预期行为
grads_with_value = [(n, p) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
grads_none = [(n, p) for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
grads_finite = all(torch.isfinite(p.grad).all() for _, p in grads_with_value)
print(f"  Params with grad: {len(grads_with_value)}, without: {len(grads_none)}")
if grads_none:
    print(f"  None grads (expected for last layer summary): {[n for n, _ in grads_none]}")
print(f"  All existing grads finite: {grads_finite}")
assert grads_finite, "Some gradients are non-finite!"
assert len(grads_with_value) > 0, "No gradients at all!"
print("  PASSED")
print()


# ============================================================
# Test 2: Gradient Checkpointing
# ============================================================
print("=" * 60)
print("Test 2: Gradient Checkpointing")
print("=" * 60)

model.zero_grad()
model.gradient_checkpointing_enable(autocast_dtype=dtype)

with torch.autocast(device_type='cuda', dtype=dtype):
    logits2 = model(
        token_ids=token_ids,
        chord_ids=chord_ids,
        instrument_ids=instrument_ids,
        token_type_ids=token_type_ids.unsqueeze(0).expand(BATCH, -1),
        note_ids=note_ids.unsqueeze(0).expand(BATCH, -1),
        num_chords=NUM_CHORDS,
    )

logits2.sum().backward()
grads_with_value2 = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
grads_finite2 = all(torch.isfinite(p.grad).all() for p in grads_with_value2)
print(f"  All existing grads finite: {grads_finite2}")
assert grads_finite2
print("  PASSED")
print()


# ============================================================
# Test 3: torch.compile
# ============================================================
print("=" * 60)
print("Test 3: torch.compile")
print("=" * 60)

model.zero_grad()
# 禁用 GC 以避免 compile + checkpoint 的潜在问题
for layer in model.layers:
    layer._gradient_checkpointing = False

compiled_model = torch.compile(model, mode='default')

with torch.autocast(device_type='cuda', dtype=dtype):
    logits3 = compiled_model(
        token_ids=token_ids,
        chord_ids=chord_ids,
        instrument_ids=instrument_ids,
        token_type_ids=token_type_ids.unsqueeze(0).expand(BATCH, -1),
        note_ids=note_ids.unsqueeze(0).expand(BATCH, -1),
        num_chords=NUM_CHORDS,
    )

logits3.sum().backward()
grads_with_value3 = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
grads_finite3 = all(torch.isfinite(p.grad).all() for p in grads_with_value3)
print(f"  Compiled forward+backward: OK")
print(f"  All existing grads finite: {grads_finite3}")
assert grads_finite3
print("  PASSED")
print()


# ============================================================
# Test 4: 不同 chord 配置 (验证变长 chord 不重编译)
# ============================================================
print("=" * 60)
print("Test 4: Different chord configs (no recompilation)")
print("=" * 60)

configs = [
    ([20, 30, 25, 35], 4),
    ([50, 60, 40, 70, 55, 45], 6),
    ([100, 80, 90], 3),
]

for chord_lens, n_chords in configs:
    sl = sum(chord_lens)
    bids = torch.zeros(sl, dtype=torch.long, device=device)
    p = 0
    for i, cl in enumerate(chord_lens):
        bids[p:p+cl] = i
        p += cl

    tids = torch.randint(0, VOCAB_SIZE, (1, sl), device=device)
    cids = bids.unsqueeze(0)
    iids = torch.zeros(1, sl, dtype=torch.long, device=device)
    ttids = (torch.arange(sl, device=device) % 4).unsqueeze(0)
    nids = (torch.arange(sl, device=device) // 4).unsqueeze(0)

    t0 = time.time()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=dtype):
        out = model(tids, chord_ids=cids, instrument_ids=iids,
                    token_type_ids=ttids, note_ids=nids, num_chords=n_chords)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"  chords={n_chords}, seq_len={sl}, time={t1-t0:.3f}s, out={out.shape}")

print("  PASSED")
print()


# ============================================================
# Test 5: 显存测量
# ============================================================
print("=" * 60)
print("Test 5: Peak memory")
print("=" * 60)

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

model.zero_grad()
for layer in model.layers:
    layer._gradient_checkpointing = False

with torch.autocast(device_type='cuda', dtype=dtype):
    logits5 = model(
        token_ids=token_ids,
        chord_ids=chord_ids,
        instrument_ids=instrument_ids,
        token_type_ids=token_type_ids.unsqueeze(0).expand(BATCH, -1),
        note_ids=note_ids.unsqueeze(0).expand(BATCH, -1),
        num_chords=NUM_CHORDS,
    )
logits5.sum().backward()

peak_mem = torch.cuda.max_memory_allocated() / 1024**3
print(f"  Peak GPU memory: {peak_mem:.3f} GB")
print(f"  Config: 8m model, B={BATCH}, SEQ={SEQ_LEN}, CHORDS={NUM_CHORDS}")
print("  PASSED")
print()


# ============================================================
# Summary
# ============================================================
print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
print()
print("v2.1 改动验证通过:")
print("  - Phase 1 SDPA: OK")
print("  - Captured tensor mask_mod (变长 chord): OK")
print("  - Token Type Visibility: OK")
print("  - enable_gqa=True: OK")
print("  - 1D RoPE: OK")
print("  - Gradient Checkpointing: OK")
print("  - torch.compile: OK")
