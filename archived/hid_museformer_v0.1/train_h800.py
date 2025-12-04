#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HID-MuseFormer H800 优化训练脚本

针对 H800 80GB 优化:
- Flash Attention 2 (显存减少 ~50%, 速度提升 2-4x)
- BF16 混合精度 (H800 原生支持)
- Gradient Checkpointing (显存换速度)
- 预处理数据加载 (.pt 文件)
- 高效 DataLoader (prefetch, pin_memory)
- torch.compile (PyTorch 2.0+)

用法:
    # 单 GPU
    python train_h800.py --data_dir processed_data/ --model_size base

    # 多 GPU (DDP)
    torchrun --nproc_per_node=8 train_h800.py --data_dir processed_data/ --model_size large

    # 多节点
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=xxx \
        train_h800.py --data_dir processed_data/ --model_size xlarge
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# 检测 Flash Attention
try:
    from flash_attn import flash_attn_func
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("[Warning] Flash Attention not available, using PyTorch SDPA")

from model import HIDMuseFormer, create_model
from data import HIDTokenizerV2, collate_fn


# ============================================================================
# 优化的 Dataset (预处理数据)
# ============================================================================

class PreprocessedDataset(Dataset):
    """
    加载预处理的 .pt 文件

    相比运行时转换，速度提升 10-100x
    """

    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 24576,
        shuffle_files: bool = True,
    ):
        self.max_seq_len = max_seq_len

        # 加载文件列表
        data_path = Path(data_path)
        if data_path.is_file():
            with open(data_path, 'r') as f:
                self.files = [line.strip() for line in f if line.strip()]
        else:
            self.files = list(data_path.glob('*.pt'))
            self.files = [str(f) for f in self.files]

        if shuffle_files:
            import random
            random.shuffle(self.files)

        print(f"Loaded {len(self.files)} preprocessed files")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = torch.load(self.files[idx], weights_only=True)

        # 截断 (如果需要)
        length = data['length']
        if length > self.max_seq_len:
            for key in ['token_ids', 'chord_ids', 'position_ids', 'instrument_ids', 'is_chord_token']:
                if key in data:
                    data[key] = data[key][:self.max_seq_len]
            data['length'] = self.max_seq_len

        return data


# ============================================================================
# 高效 Collate 函数 (动态 padding 到 batch 最大长度)
# ============================================================================

def efficient_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    高效的批次整理函数
    - 动态 padding 到 batch 内最大长度 (减少无效计算)
    - 使用 torch.nn.utils.rnn.pad_sequence
    """
    # 按长度排序 (减少 padding)
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)

    max_len = batch[0]['length']
    batch_size = len(batch)

    # 预分配张量
    token_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    chord_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
    position_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
    instrument_ids = torch.full((batch_size, max_len), 129, dtype=torch.long)
    is_chord_token = torch.zeros(batch_size, max_len, dtype=torch.bool)
    key_padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        length = item['length']
        token_ids[i, :length] = item['token_ids']
        labels[i, :length] = item['token_ids']
        chord_ids[i, :length] = item['chord_ids']
        position_ids[i, :length] = item['position_ids']
        instrument_ids[i, :length] = item['instrument_ids']
        is_chord_token[i, :length] = item['is_chord_token']
        key_padding_mask[i, :length] = False
        lengths[i] = length

    return {
        'token_ids': token_ids,
        'labels': labels,
        'chord_ids': chord_ids,
        'position_ids': position_ids,
        'instrument_ids': instrument_ids,
        'is_chord_token': is_chord_token,
        'key_padding_mask': key_padding_mask,
        'lengths': lengths,
    }


# ============================================================================
# 分布式训练工具
# ============================================================================

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


# ============================================================================
# 学习率调度器
# ============================================================================

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """
    余弦退火 + 线性预热
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# H800 Trainer
# ============================================================================

class H800Trainer:
    """
    H800 优化训练器
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.rank, self.world_size, self.local_rank = setup_distributed()
        self.is_main = self.rank == 0

        # 设备
        self.device = torch.device(f'cuda:{self.local_rank}')

        # 模型
        self.model = model.to(self.device)

        # Gradient Checkpointing
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            if self.is_main:
                print("[✓] Gradient Checkpointing enabled")

        # torch.compile (PyTorch 2.0+)
        if args.compile and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode='reduce-overhead')
            if self.is_main:
                print("[✓] torch.compile enabled")

        # DDP
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=False,
            )

        # 数据加载器 (优化配置)
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if self.world_size > 1 else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=efficient_collate_fn,
            pin_memory=True,
            prefetch_factor=4 if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0,
        )

        if val_dataset is not None:
            val_sampler = DistributedSampler(val_dataset, shuffle=False) if self.world_size > 1 else None
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=args.num_workers,
                collate_fn=efficient_collate_fn,
                pin_memory=True,
            )
        else:
            self.val_loader = None

        # 优化器 (使用 fused Adam)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
            fused=True,  # H800 优化
        )

        # 学习率调度器
        total_steps = len(self.train_loader) * args.epochs // args.gradient_accumulation_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps,
        )

        # BF16 (H800 原生支持, 比 FP16 更稳定)
        self.scaler = None  # BF16 不需要 GradScaler
        self.autocast_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

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
                    name=args.run_name or f"hid-museformer-h800-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    config=vars(args),
                )

    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        start_time = time.time()

        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(self.epoch)

        self.optimizer.zero_grad(set_to_none=True)  # 更高效

        for batch_idx, batch in enumerate(self.train_loader):
            # 移动数据到设备 (non_blocking)
            token_ids = batch['token_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            chord_ids = batch['chord_ids'].to(self.device, non_blocking=True)
            instrument_ids = batch['instrument_ids'].to(self.device, non_blocking=True)
            is_chord_token = batch['is_chord_token'].to(self.device, non_blocking=True)
            key_padding_mask = batch['key_padding_mask'].to(self.device, non_blocking=True)

            # 前向传播 (BF16/FP16)
            with torch.autocast(device_type='cuda', dtype=self.autocast_dtype):
                logits = self.model(
                    token_ids,
                    chord_ids=chord_ids,
                    instrument_ids=instrument_ids,
                    is_chord_token=is_chord_token,
                    key_padding_mask=key_padding_mask,
                )

                # 计算损失
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss / self.args.gradient_accumulation_steps

            # 反向传播
            loss.backward()

            batch_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * self.args.gradient_accumulation_steps
            total_tokens += batch_tokens
            num_batches += 1

            # 梯度累积
            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                # 优化步骤
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # 日志
                if self.is_main and self.global_step % self.args.log_interval == 0:
                    elapsed = time.time() - start_time
                    avg_loss = total_loss / num_batches
                    tokens_per_sec = total_tokens / elapsed
                    lr = self.scheduler.get_last_lr()[0]

                    print(f"Step {self.global_step} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {lr:.2e} | "
                          f"Tokens/s: {tokens_per_sec:.0f}")

                    if self.args.use_wandb:
                        import wandb
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': lr,
                            'train/tokens_per_sec': tokens_per_sec,
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
            chord_ids = batch['chord_ids'].to(self.device)
            instrument_ids = batch['instrument_ids'].to(self.device)
            is_chord_token = batch['is_chord_token'].to(self.device)
            key_padding_mask = batch['key_padding_mask'].to(self.device)

            with torch.autocast(device_type='cuda', dtype=self.autocast_dtype):
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
            print("\n" + "=" * 60)
            print("HID-MuseFormer H800 训练")
            print("=" * 60)
            print(f"设备: {self.device}")
            print(f"世界大小: {self.world_size}")
            print(f"批大小 (per GPU): {self.args.batch_size}")
            print(f"梯度累积: {self.args.gradient_accumulation_steps}")
            print(f"有效批大小: {self.args.batch_size * self.world_size * self.args.gradient_accumulation_steps}")
            print(f"Epochs: {self.args.epochs}")
            print(f"学习率: {self.args.learning_rate}")
            print(f"混合精度: {'BF16' if self.args.bf16 else ('FP16' if self.args.fp16 else 'FP32')}")
            print(f"Flash Attention: {'Yes' if HAS_FLASH_ATTN else 'No (using SDPA)'}")
            print("=" * 60 + "\n")

        for epoch in range(self.args.epochs):
            self.epoch = epoch

            if self.is_main:
                print(f"\n{'='*50}")
                print(f"Epoch {epoch + 1}/{self.args.epochs}")
                print(f"{'='*50}")

            epoch_start = time.time()

            # 训练
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            epoch_time = time.time() - epoch_start

            if self.is_main:
                print(f"Epoch {epoch + 1} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Time: {epoch_time:.1f}s", end='')

                if val_metrics:
                    print(f" | Val Loss: {val_metrics['val_loss']:.4f}", end='')

                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint('best')
                        print(" (best)", end='')

                print()

                if self.args.use_wandb:
                    import wandb
                    wandb.log({
                        'epoch': epoch + 1,
                        'epoch_time': epoch_time,
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

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"检查点已加载: {path}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='HID-MuseFormer H800 训练')

    # 数据
    parser.add_argument('--data_dir', type=str, required=True, help='预处理数据目录')
    parser.add_argument('--val_split', type=float, default=0.05, help='验证集比例')
    parser.add_argument('--max_seq_len', type=int, default=24576, help='最大序列长度 (H800: 24K+)')

    # 模型
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large', 'xlarge'], help='模型大小')

    # 训练
    parser.add_argument('--epochs', type=int, default=50, help='训练 epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小 (per GPU)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='梯度累积步数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='预热步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪')

    # H800 优化
    parser.add_argument('--bf16', action='store_true', default=True, help='使用 BF16 (推荐)')
    parser.add_argument('--fp16', action='store_true', help='使用 FP16')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True, help='使用梯度检查点')
    parser.add_argument('--compile', action='store_true', help='使用 torch.compile')

    # 输出
    parser.add_argument('--output_dir', type=str, default='./outputs_h800', help='输出目录')
    parser.add_argument('--log_interval', type=int, default=50, help='日志间隔')
    parser.add_argument('--save_interval', type=int, default=2000, help='保存间隔')

    # 其他
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载工作进程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')

    # WandB
    parser.add_argument('--use_wandb', action='store_true', help='使用 WandB 记录')
    parser.add_argument('--wandb_project', type=str, default='hid-museformer-h800', help='WandB 项目名')
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

    # 加载预处理数据集
    print("加载预处理数据集...")
    full_dataset = PreprocessedDataset(
        args.data_dir,
        max_seq_len=args.max_seq_len,
    )

    # 划分训练/验证集
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

    # 创建模型
    print(f"\n创建模型 (size={args.model_size})...")
    model = create_model(
        vocab_size=tokenizer.vocab_size,
        model_size=args.model_size,
        max_seq_len=args.max_seq_len,
        chord_start_id=tokenizer.chord_start_id,
        chord_end_id=tokenizer.chord_end_id,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # 创建训练器
    trainer = H800Trainer(model, train_dataset, val_dataset, args)

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
