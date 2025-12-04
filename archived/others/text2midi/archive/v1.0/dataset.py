"""
MidiCaps 数据集处理
将 MidiCaps JSON 数据转换为训练格式
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Optional
import random


class MidiCapsDataset(Dataset):
    """
    MidiCaps 数据集

    数据格式:
    {
        "location": "lmd_full/1/xxx.mid",
        "caption": "A short classical piece...",
        "genre": ["classical", "electronic"],
        "mood": ["film", "melodic"],
        "key": "A major",
        "time_signature": "3/4",
        "tempo": 104,
        ...
    }
    """

    def __init__(
        self,
        json_path: str,
        midi_token_dir: str,
        tokenizer,
        midi_vocab: Dict[str, int],
        max_text_len: int = 256,
        max_midi_len: int = 2048,
        use_thinking: bool = True,
    ):
        """
        Args:
            json_path: MidiCaps JSON 文件路径
            midi_token_dir: MIDI token 文件目录
            tokenizer: 文本 tokenizer
            midi_vocab: MIDI token 到 id 的映射
            max_text_len: 最大文本长度
            max_midi_len: 最大 MIDI 长度
            use_thinking: 是否使用思维链
        """
        self.tokenizer = tokenizer
        self.midi_vocab = midi_vocab
        self.max_text_len = max_text_len
        self.max_midi_len = max_midi_len
        self.use_thinking = use_thinking

        # 特殊 token
        self.bos_id = midi_vocab.get('<s>', midi_vocab.get('</s>', 0))
        self.eos_id = midi_vocab.get('</s>', midi_vocab.get('<s>', 1))
        self.pad_id = midi_vocab.get('<pad>', 0)

        # 加载数据
        print(f"加载数据集: {json_path}")
        with open(json_path, 'r') as f:
            self.data = [json.loads(line) for line in f]
        print(f"  加载了 {len(self.data)} 条数据")

        self.midi_token_dir = midi_token_dir

    def __len__(self):
        return len(self.data)

    def format_thinking(self, item: Dict) -> str:
        """格式化思维链"""
        parts = []

        # 流派
        if item.get('genre'):
            parts.append(f"流派: {', '.join(item['genre'])}")

        # 情绪
        if item.get('mood'):
            parts.append(f"情绪: {', '.join(item['mood'][:5])}")

        # 调性
        if item.get('key'):
            parts.append(f"调性: {item['key']}")

        # 拍号
        if item.get('time_signature'):
            parts.append(f"拍号: {item['time_signature']}")

        # 速度
        if item.get('tempo'):
            tempo_word = item.get('tempo_word', '')
            parts.append(f"速度: {item['tempo']} BPM ({tempo_word})")

        # 时长
        if item.get('duration'):
            duration_word = item.get('duration_word', '')
            parts.append(f"时长: {item['duration']}秒 ({duration_word})")

        # 乐器
        if item.get('instrument_summary'):
            parts.append(f"乐器: {', '.join(item['instrument_summary'])}")

        # 和弦
        if item.get('chord_summary'):
            parts.append(f"和弦: {' - '.join(item['chord_summary'][:8])}")

        return '\n'.join(parts)

    def format_input(self, item: Dict) -> str:
        """格式化输入文本"""
        caption = item.get('caption', '')

        if self.use_thinking:
            thinking = self.format_thinking(item)
            return f"{caption}\n\n<think>\n{thinking}\n</think>"
        else:
            return caption

    def load_midi_tokens(self, location: str) -> Optional[List[int]]:
        """加载 MIDI token 文件"""
        # 从 location 提取文件名
        # location: "lmd_full/1/xxx.mid" -> "xxx.txt"
        midi_name = os.path.basename(location).replace('.mid', '.txt')
        token_path = os.path.join(self.midi_token_dir, midi_name)

        if not os.path.exists(token_path):
            return None

        with open(token_path, 'r') as f:
            token_str = f.read().strip()

        # 转换为 token ids
        tokens = token_str.split()
        token_ids = []
        for t in tokens:
            if t in self.midi_vocab:
                token_ids.append(self.midi_vocab[t])
            else:
                # 未知 token 跳过
                pass

        return token_ids

    def __getitem__(self, idx):
        item = self.data[idx]

        # 格式化输入文本
        text = self.format_input(item)

        # 加载 MIDI tokens
        midi_tokens = self.load_midi_tokens(item.get('location', ''))

        if midi_tokens is None:
            # 如果没有对应的 MIDI 文件，返回空
            midi_tokens = [self.bos_id, self.eos_id]

        # 添加 BOS/EOS
        midi_tokens = [self.bos_id] + midi_tokens[:self.max_midi_len - 2] + [self.eos_id]

        return {
            'text': text,
            'midi_tokens': midi_tokens,
            'location': item.get('location', ''),
        }


def collate_fn(batch, tokenizer, midi_pad_id, max_text_len=256, max_midi_len=2048):
    """
    DataLoader collate 函数
    """
    texts = [item['text'] for item in batch]
    midi_tokens_list = [item['midi_tokens'] for item in batch]

    # Tokenize 文本
    text_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_text_len,
        return_tensors='pt'
    )

    # Pad MIDI tokens
    max_midi_len_batch = min(max(len(t) for t in midi_tokens_list), max_midi_len)
    midi_tokens_padded = []
    midi_masks = []

    for tokens in midi_tokens_list:
        # 截断
        tokens = tokens[:max_midi_len_batch]
        # 计算 padding
        pad_len = max_midi_len_batch - len(tokens)
        # Pad
        padded = tokens + [midi_pad_id] * pad_len
        mask = [False] * len(tokens) + [True] * pad_len

        midi_tokens_padded.append(padded)
        midi_masks.append(mask)

    return {
        'input_ids': text_inputs['input_ids'],
        'attention_mask': text_inputs['attention_mask'],
        'midi_tokens': torch.LongTensor(midi_tokens_padded),
        'midi_mask': torch.BoolTensor(midi_masks),
    }


def load_midi_vocab(dict_path: str) -> Dict[str, int]:
    """加载 MIDI 词表"""
    vocab = {}
    # 特殊 token
    vocab['<pad>'] = 0
    vocab['<s>'] = 1  # BOS
    vocab['</s>'] = 2  # EOS
    vocab['<unk>'] = 3

    with open(dict_path, 'r') as f:
        for i, line in enumerate(f, start=4):
            token = line.strip().split()[0]
            vocab[token] = i

    return vocab


def create_dataloader(
    json_path: str,
    midi_token_dir: str,
    tokenizer,
    midi_vocab: Dict[str, int],
    batch_size: int = 8,
    max_text_len: int = 256,
    max_midi_len: int = 2048,
    shuffle: bool = True,
    num_workers: int = 4,
    use_thinking: bool = True,
):
    """创建 DataLoader"""
    dataset = MidiCapsDataset(
        json_path=json_path,
        midi_token_dir=midi_token_dir,
        tokenizer=tokenizer,
        midi_vocab=midi_vocab,
        max_text_len=max_text_len,
        max_midi_len=max_midi_len,
        use_thinking=use_thinking,
    )

    midi_pad_id = midi_vocab.get('<pad>', 0)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(
            b, tokenizer, midi_pad_id, max_text_len, max_midi_len
        ),
        pin_memory=True,
    )

    return dataloader


# ============================================
# 数据预处理脚本
# ============================================

def preprocess_midicaps(
    midicaps_json: str,
    midi_dir: str,
    output_token_dir: str,
    encoding_method: str = 'REMIGEN2'
):
    """
    预处理 MidiCaps 数据集

    1. 读取 JSON
    2. 将每个 MIDI 文件转换为 token
    3. 保存 token 文件

    Args:
        midicaps_json: MidiCaps JSON 文件
        midi_dir: MIDI 文件根目录
        output_token_dir: 输出 token 目录
        encoding_method: MIDI 编码方法
    """
    import midiprocessor as mp

    os.makedirs(output_token_dir, exist_ok=True)

    print(f"预处理 MidiCaps 数据集")
    print(f"  JSON: {midicaps_json}")
    print(f"  MIDI 目录: {midi_dir}")
    print(f"  输出目录: {output_token_dir}")

    # 加载 JSON
    with open(midicaps_json, 'r') as f:
        data = [json.loads(line) for line in f]

    print(f"  共 {len(data)} 条数据")

    # 创建编码器
    enc = mp.MidiEncoder(encoding_method)

    success = 0
    failed = 0

    for i, item in enumerate(data):
        if i % 1000 == 0:
            print(f"  处理进度: {i}/{len(data)}")

        location = item.get('location', '')
        midi_path = os.path.join(midi_dir, location)

        if not os.path.exists(midi_path):
            failed += 1
            continue

        try:
            # 编码
            tokens = enc.encode_file(midi_path)
            token_str = ' '.join(tokens)

            # 保存
            output_name = os.path.basename(location).replace('.mid', '.txt')
            output_path = os.path.join(output_token_dir, output_name)

            with open(output_path, 'w') as f:
                f.write(token_str)

            success += 1
        except Exception as e:
            failed += 1

    print(f"\n预处理完成:")
    print(f"  成功: {success}")
    print(f"  失败: {failed}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true', help='预处理数据')
    parser.add_argument('--json', type=str, help='MidiCaps JSON 路径')
    parser.add_argument('--midi-dir', type=str, help='MIDI 文件目录')
    parser.add_argument('--output-dir', type=str, help='输出目录')

    args = parser.parse_args()

    if args.preprocess:
        preprocess_midicaps(
            args.json,
            args.midi_dir,
            args.output_dir
        )
