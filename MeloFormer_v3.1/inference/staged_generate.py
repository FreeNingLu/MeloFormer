#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
staged_generate.py - MeloFormer v2.1 分阶段条件生成 (Chain-of-Thought 风格)

核心思想:
    鼓 -> 贝斯 -> 伴奏 -> 旋律，每阶段基于前面轨道的完整结果生成

生成流程:
    阶段1: BOS + 元信息 -> 生成完整鼓轨道
           | 鼓轨道作为 prompt
    阶段2: [鼓] + #P33 SEP Chord -> 生成完整贝斯
           | 鼓+贝斯作为 prompt
    阶段3: [鼓+贝斯] + #P0 SEP Chord -> 生成完整钢琴
           | 继续...
    阶段4: [鼓+贝斯+钢琴] + #P25 SEP Chord -> 生成完整吉他

特点:
    - 真正的条件生成: 后轨道能"看到"前轨道的完整内容
    - 类似 CoT: 先定节奏框架，再逐层叠加和声与旋律
    - MeloFormer v2.1: Summary Token 自动聚合跨 chord 信息

与 MIDILM staged_generate 的区别:
    - 无 KV cache: 每阶段从完整序列重新 forward
    - 需要维护 chord_ids / instrument_ids / token_type_ids / note_ids
    - FlexAttention mask 每次 forward 重新构建

使用:
    python -m inference.staged_generate                    # GPU模式 (默认)
    python -m inference.staged_generate --device cpu       # CPU模式
    python -m inference.staged_generate --checkpoint /path/to/model.pt
"""

import sys
import torch
import argparse
from pathlib import Path
import random

# 添加项目根目录到 path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inference.generator import load_model, MusicGenerator


def count_notes(token_ids, tokenizer):
    """统计音符数量"""
    return sum(1 for tid in token_ids
               if tokenizer[tid].startswith('P') and tokenizer[tid][1:].isdigit())


def staged_generate(
    generator: MusicGenerator,
    tokenizer,
    instruments: dict,
    chords: list,
    tempo: int = 120,
    time_signature: str = '4/4',
    max_per_stage: int = 1024,
    gen_params: dict = None,
    verbose: bool = True,
) -> list:
    """
    分阶段条件生成

    Args:
        generator: MusicGenerator 实例
        tokenizer: HIDTokenizerV2 实例
        instruments: {'drum': 128, 'bass': 33, 'piano': 0, ...}
        chords: 和弦进行列表
        tempo: BPM
        time_signature: 拍号
        max_per_stage: 每阶段最大 token 数
        gen_params: 生成参数字典
        verbose: 是否显示详细信息

    Returns:
        完整的 token ID 列表
    """
    if gen_params is None:
        gen_params = {
            'temperature': 1.1,
            'top_k': 80,
            'top_p': 0.95,
            'use_music_penalty': True,
            'bar_loop_penalty': 0.6,
            'max_same_chord': 2,
            'pattern_penalty': 0.2,
            'block_instrument_switch': True,
            'force_chords': True,
        }

    instrument_order = list(instruments.items())
    note_counts = {}

    # 阶段 1: 第一个乐器
    stage_name, stage_inst = instrument_order[0]
    if verbose:
        print(f"\n[阶段 1/{len(instrument_order)}] 生成 {stage_name} 轨...")

    current_ids = generator.generate(
        instruments=[stage_inst],
        tempo=tempo,
        time_signature=time_signature,
        chords=chords,
        max_length=max_per_stage,
        verbose=verbose,
        **gen_params
    )
    note_counts[stage_name] = count_notes(current_ids, tokenizer)
    if verbose:
        print(f"  -> {stage_name}: {len(current_ids)} tokens, "
              f"{note_counts[stage_name]} 音符")

    # 后续阶段: 基于前序轨道续写
    for stage_idx in range(1, len(instrument_order)):
        stage_name, stage_inst = instrument_order[stage_idx]
        if verbose:
            print(f"\n[阶段 {stage_idx + 1}/{len(instrument_order)}] 续写 {stage_name}...")

        prev_len = len(current_ids)

        # 构建 prompt: 前序轨道 + 新乐器标记 + SEP + 第一个和弦
        prompt = current_ids.copy()
        if stage_inst == 128:
            prompt.append(tokenizer['#D'])
        else:
            prompt.append(tokenizer[f'#P{stage_inst}'])
        prompt.append(tokenizer.sep_id)
        prompt.append(tokenizer[chords[0]])

        current_ids = generator.generate(
            prompt_ids=prompt,
            chords=chords,
            max_length=len(prompt) + max_per_stage,
            verbose=verbose,
            **gen_params
        )
        note_counts[stage_name] = count_notes(current_ids[prev_len:], tokenizer)
        if verbose:
            print(f"  -> {stage_name}: +{len(current_ids) - prev_len} tokens, "
                  f"{note_counts[stage_name]} 音符")

    return current_ids, note_counts


def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description='MeloFormer v2.1 分阶段条件生成')
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'],
                        help='设备模式: cpu 或 gpu (默认: gpu)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='检查点路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认: outputs/staged_<ckpt_name>/)')
    parser.add_argument('--tempo', type=int, default=72, help='BPM')
    parser.add_argument('--time_signature', type=str, default='4/4', help='拍号')
    parser.add_argument('--max_per_stage', type=int, default=1024,
                        help='每阶段最大 token 数')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='生成样本数量')
    parser.add_argument('--chords', type=str, nargs='+', default=None,
                        help='和弦进行 (如: Cmaj Amin Fmaj Gmaj)')
    parser.add_argument('--temperature', type=float, default=1.1, help='采样温度')
    parser.add_argument('--top_k', type=int, default=80, help='Top-K 采样')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-P 采样')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='显示详细信息')
    args = parser.parse_args()

    # 配置
    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        checkpoint = str(_ROOT / "training" / "checkpoints" / "best.pt")

    # 从检查点路径提取模型标识
    ckpt_name = Path(checkpoint).stem
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _ROOT / "outputs" / f"staged_{ckpt_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设备
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("[警告] GPU 不可用，自动切换到 CPU")

    print("=" * 70)
    print("MeloFormer v2.1 分阶段生成")
    print("=" * 70)
    print(f"模型: {Path(checkpoint).name}")
    print(f"设备: {device}")
    print("=" * 70)

    # 加载模型
    print("\n加载模型...")
    model, tokenizer = load_model(checkpoint, device)
    generator = MusicGenerator(model, tokenizer, device)
    print("模型加载完成\n")

    # 固定乐器配置
    instruments = {
        'drum': 128,
        'bass': 33,
        'piano': 0,
        'guitar': 25,
    }

    # 默认和弦进行
    if args.chords:
        chords = args.chords
    else:
        chords = [
            'Cmaj', 'Amin', 'Fmaj', 'Gmaj', 'Cmaj', 'Amin', 'Fmaj', 'Gmaj',
            'Emin', 'Amin', 'Fmaj', 'Gmaj', 'Emin', 'Amin', 'Fmaj', 'Gmaj',
            'Cmaj', 'Amin', 'Dmaj', 'Gmaj', 'Cmaj', 'Amin', 'Dmaj', 'Gmaj',
            'Cmaj', 'Amin', 'Dmaj', 'Gmaj', 'Cmaj', 'Amin', 'Dmaj', 'Gmaj',
        ]

    # 生成参数
    gen_params = {
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'use_music_penalty': True,
        'bar_loop_penalty': 0.6,
        'max_same_chord': 2,
        'pattern_penalty': 0.2,
        'block_instrument_switch': True,
        'force_chords': True,
    }

    print(f"生成参数:")
    print(f"  乐器: {' + '.join(f'{k}({v})' for k, v in instruments.items())}")
    print(f"  速度: {args.tempo} BPM")
    print(f"  每阶段最大长度: {args.max_per_stage} tokens")
    print(f"  温度: {gen_params['temperature']}, "
          f"Top-K: {gen_params['top_k']}, Top-P: {gen_params['top_p']}")
    print("=" * 70)

    # 生成
    for idx in range(1, args.num_samples + 1):
        print(f"\n{'=' * 70}")
        print(f"[{idx}/{args.num_samples}] 生成样本 {idx}")
        print(f"{'=' * 70}")

        try:
            current_ids, note_counts = staged_generate(
                generator=generator,
                tokenizer=tokenizer,
                instruments=instruments,
                chords=chords,
                tempo=args.tempo,
                time_signature=args.time_signature,
                max_per_stage=args.max_per_stage,
                gen_params=gen_params,
                verbose=args.verbose,
            )

            # 保存
            output_path = output_dir / f"sample_{idx}.mid"
            generator.tokens_to_midi(current_ids, str(output_path))

            total_notes = sum(note_counts.values())
            print(f"\n保存成功: {output_path.name}")
            print(f"  总长度: {len(current_ids)} tokens")
            note_summary = ' + '.join(
                f'{k}{v}' for k, v in note_counts.items()
            )
            print(f"  音符统计: {note_summary} = {total_notes}")

        except Exception as e:
            print(f"\n生成失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("批量生成完成!")
    print(f"{'=' * 70}")
    print(f"输出目录: {output_dir}")
    print(f"生成数量: {args.num_samples} 首")
    print("=" * 70)


if __name__ == '__main__':
    main()
