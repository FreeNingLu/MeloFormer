#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HID-MuseFormer 生成脚本

支持：
- 无条件生成
- 续写 (给定开头)
- 条件生成 (指定和弦进行、乐器等)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Dict

import torch
import torch.nn.functional as F
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

from model import HIDMuseFormer, create_model
from data import HIDTokenizerV2


class MusicGenerator:
    """
    音乐生成器
    """

    def __init__(
        self,
        model: HIDMuseFormer,
        tokenizer: HIDTokenizerV2,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None,
        max_length: int = 2048,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        instruments: Optional[List[int]] = None,
        tempo: int = 120,
        time_signature: str = '4/4',
    ) -> List[int]:
        """
        生成音乐

        Args:
            prompt: 文本格式的提示
            prompt_ids: token ID 格式的提示
            max_length: 最大生成长度
            temperature: 温度
            top_k: Top-K 采样
            top_p: Top-P 采样
            instruments: 要使用的乐器列表
            tempo: BPM
            time_signature: 拍号

        Returns:
            生成的 token IDs
        """
        # 构建初始 prompt
        if prompt_ids is None:
            prompt_ids = self._build_prompt(
                prompt=prompt,
                instruments=instruments,
                tempo=tempo,
                time_signature=time_signature,
            )

        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # 初始化状态
        generated = prompt_tensor.clone()
        current_chord_ids = torch.zeros_like(generated)
        current_inst_ids = torch.full_like(generated, 129)  # 129 = global

        # 解析 prompt 中的乐器信息
        for i, tid in enumerate(prompt_ids):
            token = self.tokenizer[tid]
            if token.startswith('#P'):
                inst_id = int(token[2:])
                current_inst_ids[0, i:] = inst_id
            elif token == '#D':
                current_inst_ids[0, i:] = 128  # drums

        # 生成
        for step in range(max_length - len(prompt_ids)):
            # 前向传播
            logits = self.model(
                generated,
                chord_ids=current_chord_ids,
                instrument_ids=current_inst_ids,
            )

            # 获取最后位置的 logits
            next_logits = logits[0, -1, :] / temperature

            # Top-K 过滤
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[-1]] = float('-inf')

            # Top-P 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')

            # 采样
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 添加到序列
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            # 更新状态
            next_token_str = self.tokenizer[next_token.item()]

            # 更新乐器 ID
            new_inst_id = current_inst_ids[0, -1].clone()
            if next_token_str.startswith('#P'):
                new_inst_id = int(next_token_str[2:])
            elif next_token_str == '#D':
                new_inst_id = 128

            current_inst_ids = torch.cat([
                current_inst_ids,
                torch.tensor([[new_inst_id]], device=self.device)
            ], dim=1)

            # 更新和弦 ID
            new_chord_id = current_chord_ids[0, -1].clone()
            if self.tokenizer.is_chord_token(next_token.item()):
                new_chord_id = new_chord_id + 1

            current_chord_ids = torch.cat([
                current_chord_ids,
                torch.tensor([[new_chord_id]], device=self.device)
            ], dim=1)

            # 检查 EOS
            if next_token.item() == self.tokenizer.eos_id:
                break

            # 进度显示
            if step > 0 and step % 100 == 0:
                print(f"  已生成 {step} tokens...")

        return generated[0].tolist()

    def _build_prompt(
        self,
        prompt: Optional[str] = None,
        instruments: Optional[List[int]] = None,
        tempo: int = 120,
        time_signature: str = '4/4',
    ) -> List[int]:
        """构建初始 prompt"""
        token_ids = []

        # BOS
        token_ids.append(self.tokenizer.bos_id)

        # Tempo
        bpm_bin = self.tokenizer.quantize_tempo(tempo)
        token_ids.append(self.tokenizer[f'BPM_{bpm_bin}'])

        # Time signature
        ts_token = f'TS_{time_signature}'
        if ts_token in self.tokenizer.token2id:
            token_ids.append(self.tokenizer[ts_token])

        # 指定的乐器
        if instruments:
            for inst in instruments:
                if inst == 128:
                    token_ids.append(self.tokenizer['#D'])
                else:
                    token_ids.append(self.tokenizer[f'#P{inst}'])
                token_ids.append(self.tokenizer.sep_id)

        return token_ids

    def tokens_to_midi(
        self,
        token_ids: List[int],
        output_path: str,
        ticks_per_beat: int = 480,
    ) -> MidiFile:
        """
        将 token IDs 转换为 MIDI 文件

        Args:
            token_ids: token ID 列表
            output_path: 输出路径
            ticks_per_beat: 每拍 tick 数

        Returns:
            MidiFile 对象
        """
        midi = MidiFile(ticks_per_beat=ticks_per_beat)

        # 解析 tokens
        current_track = None
        current_instrument = 0
        current_tick = 0
        current_position = 0  # 当前和弦内位置 (0-15)
        chord_tick = 0  # 当前和弦开始 tick
        tempo_bpm = 120
        time_sig_num = 4
        time_sig_denom = 4

        beats_per_chord = 2
        ticks_per_chord = ticks_per_beat * beats_per_chord

        # 按乐器分组音符
        tracks_data: Dict[int, List[Dict]] = {}

        for tid in token_ids:
            token = self.tokenizer[tid]

            # 跳过特殊 token
            if token in ['PAD', 'BOS', 'EOS', 'MASK', 'UNK']:
                continue

            # Tempo
            if token.startswith('BPM_'):
                bpm_bin = int(token[4:])
                tempo_bpm = self.tokenizer.dequantize_tempo(bpm_bin)
                continue

            # Time signature
            if token.startswith('TS_'):
                ts = token[3:]
                parts = ts.split('/')
                if len(parts) == 2:
                    time_sig_num = int(parts[0])
                    time_sig_denom = int(parts[1])
                continue

            # 乐器标记
            if token.startswith('#P'):
                current_instrument = int(token[2:])
                if current_instrument not in tracks_data:
                    tracks_data[current_instrument] = []
                continue

            if token == '#D':
                current_instrument = 128  # drums channel 9
                if current_instrument not in tracks_data:
                    tracks_data[current_instrument] = []
                continue

            # SEP
            if token == 'SEP':
                chord_tick = 0
                continue

            # 和弦 token (作为时间标记)
            if self.tokenizer.is_chord_token(tid):
                chord_tick += ticks_per_chord
                current_position = 0
                continue

            # Position token
            if token.startswith('T') and token[1:].isdigit():
                current_position = int(token[1:])
                continue

            # Pitch token
            if token.startswith('P') and token[1:].isdigit():
                pitch = int(token[1:])

                # 计算绝对时间
                pos_tick = chord_tick + (current_position * ticks_per_chord) // 16

                if current_instrument in tracks_data:
                    tracks_data[current_instrument].append({
                        'pitch': pitch,
                        'start': pos_tick,
                        'duration': ticks_per_beat // 2,  # 默认八分音符
                        'velocity': 80,
                    })
                continue

        # 创建 MIDI tracks
        # 元数据 track
        meta_track = MidiTrack()
        midi.tracks.append(meta_track)

        # Tempo
        tempo_us = int(60_000_000 / tempo_bpm)
        meta_track.append(MetaMessage('set_tempo', tempo=tempo_us, time=0))

        # Time signature
        meta_track.append(MetaMessage(
            'time_signature',
            numerator=time_sig_num,
            denominator=time_sig_denom,
            time=0
        ))

        # 为每个乐器创建 track
        for inst_id, notes in tracks_data.items():
            if not notes:
                continue

            track = MidiTrack()
            midi.tracks.append(track)

            # 设置乐器
            channel = 9 if inst_id == 128 else min(inst_id, 15)
            if inst_id != 128:
                track.append(Message('program_change', program=inst_id, channel=channel, time=0))

            # 排序音符
            notes.sort(key=lambda x: x['start'])

            # 转换为 MIDI 事件
            events = []
            for note in notes:
                events.append({
                    'time': note['start'],
                    'type': 'note_on',
                    'note': note['pitch'],
                    'velocity': note['velocity'],
                    'channel': channel,
                })
                events.append({
                    'time': note['start'] + note['duration'],
                    'type': 'note_off',
                    'note': note['pitch'],
                    'velocity': 0,
                    'channel': channel,
                })

            events.sort(key=lambda x: (x['time'], x['type'] == 'note_on'))

            # 转换为 delta time
            current_time = 0
            for event in events:
                delta = event['time'] - current_time
                current_time = event['time']

                if event['type'] == 'note_on':
                    track.append(Message(
                        'note_on',
                        note=event['note'],
                        velocity=event['velocity'],
                        channel=event['channel'],
                        time=delta
                    ))
                else:
                    track.append(Message(
                        'note_off',
                        note=event['note'],
                        velocity=0,
                        channel=event['channel'],
                        time=delta
                    ))

            # 结束
            track.append(MetaMessage('end_of_track', time=0))

        # 保存
        midi.save(output_path)
        print(f"MIDI 文件已保存: {output_path}")

        return midi


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """
    加载模型和 tokenizer
    """
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get('args', {})

    # 创建 tokenizer
    tokenizer = HIDTokenizerV2()

    # 创建模型
    model = create_model(
        vocab_size=tokenizer.vocab_size,
        model_size=args.get('model_size', 'base'),
        max_seq_len=args.get('max_seq_len', 2048),
        chord_start_id=tokenizer.chord_start_id,
        chord_end_id=tokenizer.chord_end_id,
    )

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='HID-MuseFormer 音乐生成')

    # 模型
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')

    # 生成参数
    parser.add_argument('--max_length', type=int, default=1024, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度')
    parser.add_argument('--top_k', type=int, default=50, help='Top-K 采样')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-P 采样')

    # 音乐参数
    parser.add_argument('--tempo', type=int, default=120, help='BPM')
    parser.add_argument('--time_signature', type=str, default='4/4', help='拍号')
    parser.add_argument('--instruments', type=int, nargs='+', default=[0],
                        help='乐器列表 (MIDI program numbers, 128=drums)')

    # 输出
    parser.add_argument('--output', type=str, default='./generated.mid', help='输出文件路径')
    parser.add_argument('--num_samples', type=int, default=1, help='生成样本数')

    # 设备
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')

    return parser.parse_args()


def main():
    args = parse_args()

    print("HID-MuseFormer 音乐生成")
    print("=" * 50)

    device = torch.device(args.device)
    print(f"设备: {device}")

    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    model, tokenizer = load_model(args.checkpoint, device)

    # 创建生成器
    generator = MusicGenerator(model, tokenizer, device)

    # 生成
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_samples):
        print(f"\n生成样本 {i + 1}/{args.num_samples}...")

        # 生成 tokens
        token_ids = generator.generate(
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            instruments=args.instruments,
            tempo=args.tempo,
            time_signature=args.time_signature,
        )

        print(f"  生成了 {len(token_ids)} tokens")

        # 转换为 MIDI
        output_path = args.output if args.num_samples == 1 else \
            str(Path(args.output).parent / f"{Path(args.output).stem}_{i+1}{Path(args.output).suffix}")

        generator.tokens_to_midi(token_ids, output_path)

    print("\n完成！")


if __name__ == '__main__':
    main()
