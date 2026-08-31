#!/usr/bin/env python3
"""v3.1 基础功能测试：模型创建 + 前向传播"""

import torch
from model import create_model

def test_model_creation():
    """测试模型创建（全关闭 vs 全开启）"""

    print("=" * 60)
    print("v3.1 基础功能测试")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # 测试配置
    configs = [
        ("全关闭 (v2.1 baseline)", False, False, False),
        ("只开 k2v2_gate", False, True, False),
        ("只开 depth_selector", False, False, True),
        ("k2v2 + depth", False, True, True),
    ]

    for name, sub_attn, k2v2, depth in configs:
        print(f"[{name}]")
        try:
            model = create_model(
                vocab_size=512,
                model_size='8m',
                max_seq_len=1024,
                max_chords=64,
                chord_start_id=256,
                chord_end_id=257,
                sub_attn_bias=sub_attn,
                k2v2_gate=k2v2,
                depth_selector=depth,
            ).to(device)

            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  ✓ 模型创建成功，参数量: {params:.2f}M")

            # 前向传播测试
            B, L = 1, 128
            token_ids = torch.randint(0, 512, (B, L), device=device)
            chord_ids = torch.zeros(B, L, dtype=torch.long, device=device)
            for i in range(L):
                chord_ids[0, i] = min(i // 8, 15)

            with torch.no_grad():
                logits = model(token_ids, chord_ids=chord_ids, num_chords=16)

            print(f"  ✓ 前向传播成功，输出shape: {logits.shape}")

        except Exception as e:
            print(f"  ✗ 失败: {e}")

        print()

if __name__ == '__main__':
    test_model_creation()
