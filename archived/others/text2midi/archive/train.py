"""
Text2MIDI 训练脚本

训练策略:
    1. 冻结 Qwen Encoder
    2. 训练 Projection Layer + MIDI Decoder
    3. 使用 Teacher Forcing
"""

import os
import sys
import argparse
import time
import math
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer

from model import Text2MIDIModel, ModelConfig
from dataset import create_dataloader, load_midi_vocab


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

        # 混合精度训练
        self.scaler = GradScaler() if config.use_amp and device != 'cpu' else None

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
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            midi_tokens = batch['midi_tokens'].to(self.device)
            midi_mask = batch['midi_mask'].to(self.device)

            # 准备输入输出
            # 输入: midi_tokens[:, :-1]
            # 目标: midi_tokens[:, 1:]
            decoder_input = midi_tokens[:, :-1]
            target = midi_tokens[:, 1:]
            decoder_mask = midi_mask[:, :-1]

            # 前向传播
            self.optimizer.zero_grad()

            if self.scaler:
                with autocast():
                    logits = self.model(
                        input_ids, attention_mask,
                        decoder_input, decoder_mask
                    )
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        target.reshape(-1)
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(
                    input_ids, attention_mask,
                    decoder_input, decoder_mask
                )
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

            # 打印进度
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
            midi_mask = batch['midi_mask'].to(self.device)

            decoder_input = midi_tokens[:, :-1]
            target = midi_tokens[:, 1:]
            decoder_mask = midi_mask[:, :-1]

            logits = self.model(
                input_ids, attention_mask,
                decoder_input, decoder_mask
            )
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
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': vars(self.config),
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

            # 训练
            train_loss = self.train_epoch(epoch)
            print(f"  训练损失: {train_loss:.4f}")

            # 验证
            if self.val_loader:
                val_loss = self.validate()
                print(f"  验证损失: {val_loss:.4f}")

                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(self.config.output_dir, 'best_model.pt'),
                        epoch, val_loss
                    )
                    print(f"  ✓ 保存最佳模型 (val_loss: {val_loss:.4f})")

            # 定期保存
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
    midi_dict_path = "data/dict.txt"

    # 模型
    qwen_model_path = "/Users/freeninglu/Desktop/MuseFormer/Qwen3-Embedding-0.6B"
    model_size = "base"  # small, base, large

    # 训练参数
    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-4
    weight_decay = 0.01
    warmup_steps = 1000
    max_text_len = 256
    max_midi_len = 2048

    # 其他
    use_amp = True
    log_interval = 100
    save_interval = 1
    seed = 42
    num_workers = 4

    # 特殊 token
    pad_id = 0

    # 输出
    output_dir = "checkpoints"


def estimate_training_time(config, num_samples, device_type='cpu'):
    """
    估算训练时间

    Args:
        config: 训练配置
        num_samples: 数据集样本数
        device_type: 设备类型 (cpu/cuda/mps)

    Returns:
        估算的训练时间 (小时)
    """
    # 每个 step 的估算时间 (秒)
    time_per_step = {
        'cpu': 2.0,      # CPU 较慢
        'mps': 0.5,      # Apple Silicon
        'cuda_v100': 0.1,  # V100 GPU
        'cuda_a100': 0.05, # A100 GPU
    }

    # 获取时间估算
    if device_type == 'cpu':
        step_time = time_per_step['cpu']
    elif device_type == 'mps':
        step_time = time_per_step['mps']
    elif device_type == 'cuda':
        step_time = time_per_step['cuda_v100']  # 假设 V100
    else:
        step_time = 1.0

    # 计算
    steps_per_epoch = num_samples // config.batch_size
    total_steps = steps_per_epoch * config.num_epochs
    total_seconds = total_steps * step_time

    return total_seconds / 3600  # 转换为小时


def main():
    parser = argparse.ArgumentParser(description='Text2MIDI 训练')

    # 数据
    parser.add_argument('--train-json', type=str, help='训练数据 JSON')
    parser.add_argument('--val-json', type=str, help='验证数据 JSON')
    parser.add_argument('--midi-token-dir', type=str, help='MIDI token 目录')
    parser.add_argument('--midi-dict', type=str, help='MIDI 字典路径')

    # 模型
    parser.add_argument('--model-size', type=str, default='base',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--qwen-path', type=str, help='Qwen 模型路径')

    # 训练
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'])

    # 输出
    parser.add_argument('--output-dir', type=str, default='checkpoints')

    # 其他
    parser.add_argument('--estimate-only', action='store_true',
                        help='仅估算训练时间')

    args = parser.parse_args()

    # 创建配置
    config = TrainConfig()

    if args.train_json:
        config.train_json = args.train_json
    if args.val_json:
        config.val_json = args.val_json
    if args.midi_token_dir:
        config.midi_token_dir = args.midi_token_dir
    if args.midi_dict:
        config.midi_dict_path = args.midi_dict
    if args.qwen_path:
        config.qwen_model_path = args.qwen_path
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.output_dir:
        config.output_dir = args.output_dir

    config.model_size = args.model_size

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
    print("Text2MIDI 训练")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"模型大小: {config.model_size}")
    print(f"批次大小: {config.batch_size}")
    print(f"训练轮数: {config.num_epochs}")
    print(f"学习率: {config.learning_rate}")

    # 加载 MIDI 词表
    print("\n加载 MIDI 词表...")
    midi_vocab = load_midi_vocab(config.midi_dict_path)
    config.pad_id = midi_vocab.get('<pad>', 0)
    print(f"  词表大小: {len(midi_vocab)}")

    # 如果只是估算时间
    if args.estimate_only:
        # 假设数据集大小
        num_samples = 100000  # MidiCaps 约 10万条

        hours = estimate_training_time(config, num_samples, str(device.type))
        print(f"\n估算训练时间:")
        print(f"  数据集大小: {num_samples} 条")
        print(f"  设备: {device}")
        print(f"  每 epoch: {hours / config.num_epochs:.1f} 小时")
        print(f"  总计: {hours:.1f} 小时 ({hours/24:.1f} 天)")
        return

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(config), f, indent=2)

    # 加载 tokenizer
    print("\n加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_path)

    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader = create_dataloader(
        config.train_json,
        config.midi_token_dir,
        tokenizer,
        midi_vocab,
        batch_size=config.batch_size,
        max_text_len=config.max_text_len,
        max_midi_len=config.max_midi_len,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = None
    if config.val_json and os.path.exists(config.val_json):
        val_loader = create_dataloader(
            config.val_json,
            config.midi_token_dir,
            tokenizer,
            midi_vocab,
            batch_size=config.batch_size,
            max_text_len=config.max_text_len,
            max_midi_len=config.max_midi_len,
            shuffle=False,
            num_workers=config.num_workers,
        )

    # 创建模型
    print("\n创建模型...")
    if config.model_size == 'small':
        model_config = ModelConfig.small()
    elif config.model_size == 'large':
        model_config = ModelConfig.large()
    else:
        model_config = ModelConfig.base()

    model_config.qwen_model_path = config.qwen_model_path
    model_config.midi_vocab_size = len(midi_vocab)

    model = Text2MIDIModel(
        qwen_model_path=model_config.qwen_model_path,
        midi_vocab_size=model_config.midi_vocab_size,
        d_model=model_config.d_model,
        nhead=model_config.nhead,
        num_decoder_layers=model_config.num_decoder_layers,
        freeze_encoder=True
    )
    model = model.to(device)

    # 统计参数
    params = model.count_parameters()
    print(f"\n模型参数:")
    print(f"  总参数: {params['total'] / 1e6:.2f}M")
    print(f"  可训练: {params['trainable'] / 1e6:.2f}M")
    print(f"  冻结: {params['frozen'] / 1e6:.2f}M")

    # 创建优化器
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
    hours = estimate_training_time(config, num_samples, str(device.type))
    print(f"\n估算训练时间: {hours:.1f} 小时")

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
