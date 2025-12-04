"""
Text2MIDI 训练脚本 (最终版)

训练策略:
    1. 冻结 Qwen Encoder (文本编码器)
    2. 冻结 MuseFormer (MIDI 解码器)
    3. 只训练连接层 (TextConditionEncoder)

这是最高效的训练方案，因为:
    - 只需要训练 ~2M 参数
    - Qwen 和 MuseFormer 的能力完全保留
    - 训练速度快，GPU 上几小时即可
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from transformers import AutoTokenizer

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Text2MIDIFinal, ModelConfig, estimate_training_time


class MidiCapsDataset(torch.utils.data.Dataset):
    """
    MidiCaps 数据集
    加载文本描述和对应的 MIDI token 序列
    """
    def __init__(
        self,
        json_path,
        midi_token_dir,
        midi_vocab,
        tokenizer,
        max_text_len=256,
        max_midi_len=2048,
        use_thinking=True,
    ):
        self.tokenizer = tokenizer
        self.midi_vocab = midi_vocab
        self.max_text_len = max_text_len
        self.max_midi_len = max_midi_len
        self.use_thinking = use_thinking
        self.midi_token_dir = midi_token_dir

        # 特殊 token
        self.bos_id = midi_vocab.get('<s>', 1)
        self.eos_id = midi_vocab.get('</s>', 2)
        self.pad_id = midi_vocab.get('<pad>', 0)

        # 加载数据
        print(f"加载数据集: {json_path}")
        with open(json_path, 'r') as f:
            self.data = [json.loads(line) for line in f]
        print(f"  加载了 {len(self.data)} 条数据")

    def format_thinking(self, item):
        """格式化思维链"""
        parts = []

        if item.get('genre'):
            parts.append(f"流派: {', '.join(item['genre'])}")
        if item.get('mood'):
            parts.append(f"情绪: {', '.join(item['mood'][:5])}")
        if item.get('key'):
            parts.append(f"调性: {item['key']}")
        if item.get('time_signature'):
            parts.append(f"拍号: {item['time_signature']}")
        if item.get('tempo'):
            parts.append(f"速度: {item['tempo']} BPM")
        if item.get('instrument_summary'):
            parts.append(f"乐器: {', '.join(item['instrument_summary'])}")

        return '\n'.join(parts)

    def format_input(self, item):
        """格式化输入文本"""
        caption = item.get('caption', '')

        if self.use_thinking:
            thinking = self.format_thinking(item)
            return f"{caption}\n\n<think>\n{thinking}\n</think>"
        else:
            return caption

    def load_midi_tokens(self, location):
        """加载 MIDI token 文件"""
        midi_name = os.path.basename(location).replace('.mid', '.txt')
        token_path = os.path.join(self.midi_token_dir, midi_name)

        if not os.path.exists(token_path):
            return None

        with open(token_path, 'r') as f:
            token_str = f.read().strip()

        tokens = token_str.split()
        token_ids = [self.midi_vocab.get(t, self.pad_id) for t in tokens]

        return token_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = self.format_input(item)
        midi_tokens = self.load_midi_tokens(item.get('location', ''))

        if midi_tokens is None:
            midi_tokens = [self.bos_id, self.eos_id]

        # 添加 BOS/EOS
        midi_tokens = [self.bos_id] + midi_tokens[:self.max_midi_len - 2] + [self.eos_id]

        return {
            'text': text,
            'midi_tokens': midi_tokens,
        }


def collate_fn(batch, tokenizer, midi_pad_id, max_text_len=256, max_midi_len=2048):
    """DataLoader collate 函数"""
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
        tokens = tokens[:max_midi_len_batch]
        pad_len = max_midi_len_batch - len(tokens)
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


def load_midi_vocab(dict_path):
    """加载 MIDI 词表 (fairseq 格式)"""
    vocab = {}
    vocab['<pad>'] = 0
    vocab['<s>'] = 1
    vocab['</s>'] = 2
    vocab['<unk>'] = 3

    with open(dict_path, 'r') as f:
        for i, line in enumerate(f, start=4):
            token = line.strip().split()[0]
            vocab[token] = i

    return vocab


class Trainer:
    """训练器"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        config,
        device='cpu'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id)

        # 日志
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            midi_tokens = batch['midi_tokens'].to(self.device)

            # 准备输入输出
            decoder_input = midi_tokens[:, :-1]
            target = midi_tokens[:, 1:]

            # 前向传播
            self.optimizer.zero_grad()

            logits = self.model(input_ids, attention_mask, decoder_input)

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"  Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f} | LR: {lr:.6f} | "
                    f"Time: {elapsed:.1f}s"
                )

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            midi_tokens = batch['midi_tokens'].to(self.device)

            decoder_input = midi_tokens[:, :-1]
            target = midi_tokens[:, 1:]

            logits = self.model(input_ids, attention_mask, decoder_input)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1)
            )

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss

    def save_checkpoint(self, path, epoch, val_loss):
        """保存检查点 (只保存训练的部分)"""
        checkpoint = {
            'epoch': epoch,
            'condition_encoder_state_dict': self.model.condition_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': {
                'condition_mode': self.model.condition_mode,
                'text_dim': self.model.text_dim,
                'museformer_dim': self.model.museformer_dim,
            },
        }
        torch.save(checkpoint, path)

    def train(self):
        """完整训练流程"""
        print("=" * 60)
        print("开始训练")
        print("=" * 60)

        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.num_epochs}")
            print("-" * 40)

            train_loss = self.train_epoch(epoch)
            print(f"  训练损失: {train_loss:.4f}")

            if self.val_loader:
                val_loss = self.validate()
                print(f"  验证损失: {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(self.config.output_dir, 'best_model.pt'),
                        epoch, val_loss
                    )
                    print(f"  ✓ 保存最佳模型 (val_loss: {val_loss:.4f})")

            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(
                    os.path.join(self.config.output_dir, f'checkpoint_epoch{epoch}.pt'),
                    epoch, val_loss if self.val_loader else train_loss
                )

        # 保存最终模型
        self.save_checkpoint(
            os.path.join(self.config.output_dir, 'final_model.pt'),
            self.config.num_epochs,
            self.val_losses[-1] if self.val_losses else self.train_losses[-1]
        )

        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)


class TrainConfig:
    """训练配置"""
    # 数据
    train_json = "data/midicaps_train.jsonl"
    val_json = "data/midicaps_val.jsonl"
    midi_token_dir = "data/midi_tokens"

    # MuseFormer 字典
    midi_dict_path = "/Users/freeninglu/Desktop/MuseFormer/muzic/museformer/data-bin/lmd6remi/dict.txt"

    # 模型
    qwen_model_path = "/Users/freeninglu/Desktop/MuseFormer/Qwen3-Embedding-0.6B"
    museformer_checkpoint = "/Users/freeninglu/Desktop/MuseFormer/muzic/museformer/checkpoints/mf-lmd6remi-1/checkpoint_best.pt"
    museformer_data_dir = "/Users/freeninglu/Desktop/MuseFormer/muzic/museformer/data-bin/lmd6remi"

    condition_mode = 'bias'
    hidden_dim = 1024

    # 训练参数
    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-4
    weight_decay = 0.01
    max_text_len = 256
    max_midi_len = 2048

    # 其他
    log_interval = 50
    save_interval = 1
    seed = 42
    num_workers = 0

    # 特殊 token
    pad_id = 0

    # 输出
    output_dir = "checkpoints"


def main():
    parser = argparse.ArgumentParser(description='Text2MIDI 训练 (使用 MuseFormer)')

    parser.add_argument('--train-json', type=str, help='训练数据 JSON')
    parser.add_argument('--val-json', type=str, help='验证数据 JSON')
    parser.add_argument('--midi-token-dir', type=str, help='MIDI token 目录')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output-dir', type=str, default='checkpoints')
    parser.add_argument('--estimate-only', action='store_true', help='仅估算训练时间')

    args = parser.parse_args()

    # 创建配置
    config = TrainConfig()

    if args.train_json:
        config.train_json = args.train_json
    if args.val_json:
        config.val_json = args.val_json
    if args.midi_token_dir:
        config.midi_token_dir = args.midi_token_dir
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.output_dir = args.output_dir

    # 选择设备
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("Text2MIDI 训练 (使用 MuseFormer)")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"批次大小: {config.batch_size}")
    print(f"训练轮数: {config.num_epochs}")
    print(f"学习率: {config.learning_rate}")

    # 如果只是估算时间
    if args.estimate_only:
        num_samples = 100000
        est = estimate_training_time(num_samples, config.batch_size, config.num_epochs, str(device.type))
        print(f"\n估算训练时间:")
        print(f"  数据集大小: {num_samples} 条")
        print(f"  设备: {device}")
        print(f"  每 epoch: {est['hours'] / config.num_epochs:.2f} 小时")
        print(f"  总计: {est['hours']:.1f} 小时 ({est['days']:.2f} 天)")
        return

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 加载 MIDI 词表
    print("\n加载 MIDI 词表...")
    midi_vocab = load_midi_vocab(config.midi_dict_path)
    config.pad_id = midi_vocab.get('<pad>', 0)
    print(f"  词表大小: {len(midi_vocab)}")

    # 加载 tokenizer
    print("\n加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_path)

    # 创建数据加载器
    print("\n创建数据加载器...")
    train_dataset = MidiCapsDataset(
        config.train_json,
        config.midi_token_dir,
        midi_vocab,
        tokenizer,
        max_text_len=config.max_text_len,
        max_midi_len=config.max_midi_len,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer, config.pad_id, config.max_text_len, config.max_midi_len),
        pin_memory=True,
    )

    val_loader = None
    if config.val_json and os.path.exists(config.val_json):
        val_dataset = MidiCapsDataset(
            config.val_json,
            config.midi_token_dir,
            midi_vocab,
            tokenizer,
            max_text_len=config.max_text_len,
            max_midi_len=config.max_midi_len,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=lambda b: collate_fn(b, tokenizer, config.pad_id, config.max_text_len, config.max_midi_len),
        )

    # 创建模型
    print("\n创建模型...")
    model = Text2MIDIFinal(
        qwen_model_path=config.qwen_model_path,
        museformer_checkpoint_path=config.museformer_checkpoint,
        museformer_data_dir=config.museformer_data_dir,
        condition_mode=config.condition_mode,
        hidden_dim=config.hidden_dim,
        freeze_encoder=True,
        freeze_decoder=True,
        device=str(device)
    )
    model = model.to(device)

    # 统计参数
    params = model.count_parameters()
    print(f"\n模型参数:")
    print(f"  总参数: {params['total'] / 1e6:.2f}M")
    print(f"  可训练: {params['trainable'] / 1e6:.2f}M (仅连接层)")
    print(f"  冻结: {params['frozen'] / 1e6:.2f}M")

    # 创建优化器 (只优化可训练参数)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 创建学习率调度器
    total_steps = len(train_loader) * config.num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=0.1,
    )

    # 估算训练时间
    num_samples = len(train_loader) * config.batch_size
    est = estimate_training_time(num_samples, config.batch_size, config.num_epochs, str(device.type))
    print(f"\n估算训练时间: {est['hours']:.1f} 小时")

    # 保存配置
    with open(os.path.join(config.output_dir, 'train_config.json'), 'w') as f:
        json.dump({
            'qwen_model_path': config.qwen_model_path,
            'museformer_checkpoint': config.museformer_checkpoint,
            'condition_mode': config.condition_mode,
            'hidden_dim': config.hidden_dim,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'learning_rate': config.learning_rate,
        }, f, indent=2)

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
