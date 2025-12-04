#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HID-MuseFormer 训练脚本

支持：
- 多 GPU 分布式训练 (DDP)
- 混合精度训练 (AMP)
- 梯度累积
- 学习率预热和衰减
- 检查点保存和恢复
- WandB 日志记录
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from model import HIDMuseFormer, create_model
from data import HIDMusicDataset, collate_fn, HIDTokenizerV2


def setup_distributed():
    """设置分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式训练"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
    """
    创建学习率调度器 (线性预热 + 余弦衰减)
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # 线性预热
            return float(current_step) / float(max(1, warmup_steps))
        # 余弦衰减
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    训练器类
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: HIDMusicDataset,
        val_dataset: Optional[HIDMusicDataset] = None,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.rank, self.world_size, self.local_rank = setup_distributed()
        self.is_main = self.rank == 0

        # 设备
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')

        # 模型
        self.model = model.to(self.device)
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        # 数据加载器
        train_sampler = DistributedSampler(train_dataset) if self.world_size > 1 else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        if val_dataset is not None:
            val_sampler = DistributedSampler(val_dataset, shuffle=False) if self.world_size > 1 else None
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )
        else:
            self.val_loader = None

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.98),
        )

        # 学习率调度器
        total_steps = len(self.train_loader) * args.epochs // args.gradient_accumulation_steps
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            warmup_steps=args.warmup_steps,
            total_steps=total_steps,
        )

        # 混合精度
        self.scaler = GradScaler() if args.fp16 else None

        # 状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # 日志
        self.log_dir = Path(args.output_dir) / 'logs'
        self.checkpoint_dir = Path(args.output_dir) / 'checkpoints'

        if self.is_main:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # WandB
            if args.use_wandb:
                import wandb
                wandb.init(
                    project=args.wandb_project,
                    name=args.run_name or f"hid-museformer-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    config=vars(args),
                )

    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(self.epoch)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # 移动数据到设备
            token_ids = batch['token_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            chord_ids = batch.get('chord_ids')
            instrument_ids = batch.get('instrument_ids')
            is_chord_token = batch.get('is_chord_token')
            key_padding_mask = batch.get('key_padding_mask')

            if chord_ids is not None:
                chord_ids = chord_ids.to(self.device)
            if instrument_ids is not None:
                instrument_ids = instrument_ids.to(self.device)
            if is_chord_token is not None:
                is_chord_token = is_chord_token.to(self.device)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.to(self.device)

            # 前向传播
            with autocast(enabled=self.args.fp16):
                logits = self.model(
                    token_ids,
                    chord_ids=chord_ids,
                    instrument_ids=instrument_ids,
                    is_chord_token=is_chord_token,
                    key_padding_mask=key_padding_mask,
                )

                # 计算损失 (shift for next token prediction)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss / self.args.gradient_accumulation_steps

            # 反向传播
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.args.gradient_accumulation_steps
            num_batches += 1

            # 梯度累积
            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                # 优化步骤
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # 日志
                if self.is_main and self.global_step % self.args.log_interval == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

                    if self.args.use_wandb:
                        import wandb
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': lr,
                            'train/step': self.global_step,
                        })

                # 保存检查点
                if self.is_main and self.global_step % self.args.save_interval == 0:
                    self.save_checkpoint(f'step_{self.global_step}')

        return {'loss': total_loss / max(num_batches, 1)}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            token_ids = batch['token_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            chord_ids = batch.get('chord_ids')
            instrument_ids = batch.get('instrument_ids')
            is_chord_token = batch.get('is_chord_token')
            key_padding_mask = batch.get('key_padding_mask')

            if chord_ids is not None:
                chord_ids = chord_ids.to(self.device)
            if instrument_ids is not None:
                instrument_ids = instrument_ids.to(self.device)
            if is_chord_token is not None:
                is_chord_token = is_chord_token.to(self.device)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.to(self.device)

            with autocast(enabled=self.args.fp16):
                logits = self.model(
                    token_ids,
                    chord_ids=chord_ids,
                    instrument_ids=instrument_ids,
                    is_chord_token=is_chord_token,
                    key_padding_mask=key_padding_mask,
                )

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # 聚合分布式结果
        if self.world_size > 1:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size

        return {'val_loss': avg_loss}

    def train(self):
        """完整训练循环"""
        if self.is_main:
            print(f"开始训练...")
            print(f"  设备: {self.device}")
            print(f"  世界大小: {self.world_size}")
            print(f"  批大小: {self.args.batch_size}")
            print(f"  梯度累积: {self.args.gradient_accumulation_steps}")
            print(f"  有效批大小: {self.args.batch_size * self.world_size * self.args.gradient_accumulation_steps}")
            print(f"  Epochs: {self.args.epochs}")
            print(f"  学习率: {self.args.learning_rate}")

        for epoch in range(self.args.epochs):
            self.epoch = epoch

            if self.is_main:
                print(f"\n{'='*50}")
                print(f"Epoch {epoch + 1}/{self.args.epochs}")
                print(f"{'='*50}")

            # 训练
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            if self.is_main:
                print(f"Epoch {epoch + 1} | Train Loss: {train_metrics['loss']:.4f}", end='')
                if val_metrics:
                    print(f" | Val Loss: {val_metrics['val_loss']:.4f}", end='')

                    # 保存最佳模型
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint('best')
                        print(" (best)", end='')

                print()

                if self.args.use_wandb:
                    import wandb
                    wandb.log({
                        'epoch': epoch + 1,
                        **{f'train/{k}': v for k, v in train_metrics.items()},
                        **{f'val/{k}': v for k, v in val_metrics.items()},
                    })

            # 保存 epoch 检查点
            if self.is_main:
                self.save_checkpoint(f'epoch_{epoch + 1}')

        if self.is_main:
            print("\n训练完成！")
            self.save_checkpoint('final')

    def save_checkpoint(self, name: str):
        """保存检查点"""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'args': vars(self.args),
        }

        path = self.checkpoint_dir / f'{name}.pt'
        torch.save(checkpoint, path)
        print(f"检查点已保存: {path}")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)

        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"检查点已加载: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description='HID-MuseFormer 训练')

    # 数据
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--train_split', type=float, default=0.95, help='训练集比例')
    parser.add_argument('--max_seq_len', type=int, default=24576, help='最大序列长度 (H800 80GB 可支持 24K+)')

    # 模型
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large', 'xlarge'], help='模型大小')
    parser.add_argument('--vocab_size', type=int, default=415, help='词汇表大小')

    # 训练
    parser.add_argument('--epochs', type=int, default=50, help='训练 epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='梯度累积步数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='预热步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--fp16', action='store_true', help='使用混合精度')

    # 输出
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--log_interval', type=int, default=100, help='日志间隔')
    parser.add_argument('--save_interval', type=int, default=5000, help='保存间隔')

    # 其他
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载工作进程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')

    # WandB
    parser.add_argument('--use_wandb', action='store_true', help='使用 WandB 记录')
    parser.add_argument('--wandb_project', type=str, default='hid-museformer', help='WandB 项目名')
    parser.add_argument('--run_name', type=str, default=None, help='运行名称')

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建 tokenizer
    tokenizer = HIDTokenizerV2()

    # 更新词汇表大小
    args.vocab_size = tokenizer.vocab_size

    # 加载数据集
    print("加载数据集...")
    data_dir = Path(args.data_dir)

    # 获取所有 MIDI 文件
    midi_files = list(data_dir.glob('**/*.mid')) + list(data_dir.glob('**/*.midi'))
    print(f"找到 {len(midi_files)} 个 MIDI 文件")

    if len(midi_files) == 0:
        print("错误: 没有找到 MIDI 文件")
        return

    # 划分训练/验证集
    num_train = int(len(midi_files) * args.train_split)
    train_files = midi_files[:num_train]
    val_files = midi_files[num_train:] if num_train < len(midi_files) else None

    print(f"训练集: {len(train_files)} 文件")
    if val_files:
        print(f"验证集: {len(val_files)} 文件")

    # 创建数据集
    train_dataset = HIDMusicDataset(
        midi_files=train_files,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )

    val_dataset = None
    if val_files:
        val_dataset = HIDMusicDataset(
            midi_files=val_files,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
        )

    # 创建模型
    print(f"创建模型 (size={args.model_size})...")
    model = create_model(
        vocab_size=args.vocab_size,
        model_size=args.model_size,
        max_seq_len=args.max_seq_len,
        chord_start_id=tokenizer.chord_start_id,
        chord_end_id=tokenizer.chord_end_id,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # 创建训练器
    trainer = Trainer(model, train_dataset, val_dataset, args)

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
