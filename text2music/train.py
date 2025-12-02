#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text2Summary 训练脚本

训练 Diffusion Bridge (Flow Matching) 模型，
将文本描述映射到 HID-MeloFormer 的 Summary Token。

用法:
    # 单 GPU
    python train.py --config configs/default.yaml

    # 多 GPU (DDP)
    torchrun --nproc_per_node=4 train.py --config configs/default.yaml
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from models.text2summary import Text2SummaryModel
from data.dataset import MIDICapsDataset, MIDICapsCollator


def setup_distributed():
    """设置分布式训练"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    """清理分布式"""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_default_config() -> Dict[str, Any]:
    """创建默认配置"""
    return {
        # 数据
        'data': {
            'train_path': 'MIDICAPS/train_augmented.jsonl',
            'val_path': 'MIDICAPS/val.jsonl',
            'summary_dir': 'summary_tokens/',
            'max_text_length': 512,
            'max_bars': 64,
        },

        # 模型
        'model': {
            'text_encoder': 'Qwen/Qwen3-Embedding-0.6B',
            'summary_dim': 512,
            'bridge_hidden_dim': 1024,
            'bridge_num_layers': 6,
            'freeze_text_encoder': True,
            'use_transformer_bridge': False,
        },

        # 训练
        'training': {
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'warmup_epochs': 2,
            'grad_clip': 1.0,
            'cfg_dropout': 0.1,
            'use_amp': True,
        },

        # 日志
        'logging': {
            'log_interval': 100,
            'save_interval': 1,
            'output_dir': 'outputs/text2summary',
        },
    }


class Trainer:
    """训练器"""

    def __init__(
        self,
        config: Dict[str, Any],
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = rank == 0

        self.device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

        # 创建输出目录
        self.output_dir = Path(config['logging']['output_dir'])
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # 保存配置
            with open(self.output_dir / 'config.yaml', 'w') as f:
                yaml.dump(config, f)

        # 初始化组件
        self._init_tokenizer()
        self._init_model()
        self._init_data()
        self._init_optimizer()

        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

    def _init_tokenizer(self):
        """初始化 tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['text_encoder'],
            trust_remote_code=True,
        )

    def _init_model(self):
        """初始化模型"""
        model_config = self.config['model']

        self.model = Text2SummaryModel(
            text_encoder_name=model_config['text_encoder'],
            summary_dim=model_config['summary_dim'],
            bridge_hidden_dim=model_config['bridge_hidden_dim'],
            bridge_num_layers=model_config['bridge_num_layers'],
            freeze_text_encoder=model_config['freeze_text_encoder'],
            use_transformer_bridge=model_config['use_transformer_bridge'],
        )

        self.model.to(self.device)

        # DDP 包装
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        # 统计参数
        if self.is_main:
            total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
            print(f"Model: {total_params:.2f}M total, {trainable_params:.2f}M trainable")

    def _init_data(self):
        """初始化数据"""
        data_config = self.config['data']

        # 训练集
        self.train_dataset = MIDICapsDataset(
            data_path=data_config['train_path'],
            summary_dir=data_config['summary_dir'],
            tokenizer=self.tokenizer,
            max_text_length=data_config['max_text_length'],
            max_bars=data_config['max_bars'],
            filter_missing=True,
        )

        # 验证集 (可选)
        self.val_dataset = None
        if 'val_path' in data_config and os.path.exists(data_config['val_path']):
            self.val_dataset = MIDICapsDataset(
                data_path=data_config['val_path'],
                summary_dir=data_config['summary_dir'],
                tokenizer=self.tokenizer,
                max_text_length=data_config['max_text_length'],
                max_bars=data_config['max_bars'],
                filter_missing=True,
            )

        # Sampler
        train_sampler = None
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )

        # Collator
        collator = MIDICapsCollator(tokenizer=self.tokenizer)

        # DataLoader
        train_config = self.config['training']
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=collator,
        )

        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=train_config['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=collator,
            )
        else:
            self.val_loader = None

        if self.is_main:
            print(f"Train samples: {len(self.train_dataset)}")
            if self.val_dataset:
                print(f"Val samples: {len(self.val_dataset)}")

    def _init_optimizer(self):
        """初始化优化器"""
        train_config = self.config['training']

        # 只优化可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = AdamW(
            trainable_params,
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
        )

        # 学习率调度器
        total_steps = len(self.train_loader) * train_config['epochs']
        warmup_steps = len(self.train_loader) * train_config['warmup_epochs']

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        # AMP
        self.scaler = torch.GradScaler() if train_config['use_amp'] else None

    def train_epoch(self) -> float:
        """训练一个 epoch"""
        self.model.train()
        train_config = self.config['training']

        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, disable=not self.is_main, desc=f"Epoch {self.epoch}")

        for batch in pbar:
            self.optimizer.zero_grad()

            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            summary_tokens = batch['summary_tokens'].to(self.device)

            # 前向传播
            if self.scaler is not None:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        summary_tokens=summary_tokens,
                        cfg_dropout=train_config['cfg_dropout'],
                    )
                    loss = outputs['loss']

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_config['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    summary_tokens=summary_tokens,
                    cfg_dropout=train_config['cfg_dropout'],
                )
                loss = outputs['loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_config['grad_clip'])
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1

            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
            })

            # 日志
            if self.is_main and self.global_step % self.config['logging']['log_interval'] == 0:
                avg_loss = total_loss / num_batches
                print(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={self.scheduler.get_last_lr()[0]:.2e}")

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> float:
        """验证"""
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, disable=not self.is_main, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            summary_tokens = batch['summary_tokens'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                summary_tokens=summary_tokens,
                cfg_dropout=0.0,  # 验证时不使用 dropout
            )

            total_loss += outputs['loss'].item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        if not self.is_main:
            return

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.output_dir / filename)
        print(f"Saved checkpoint: {self.output_dir / filename}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        if self.is_main:
            print(f"Loaded checkpoint from epoch {self.epoch}")

    def train(self):
        """完整训练"""
        train_config = self.config['training']

        for epoch in range(self.epoch, train_config['epochs']):
            self.epoch = epoch

            # 设置 epoch (用于 sampler)
            if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            # 训练
            train_loss = self.train_epoch()

            # 验证
            val_loss = self.validate()

            if self.is_main:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

                # 保存最佳模型
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint('best.pt')

                # 定期保存
                if (epoch + 1) % self.config['logging']['save_interval'] == 0:
                    self.save_checkpoint(f'epoch_{epoch + 1}.pt')

        # 保存最终模型
        if self.is_main:
            self.save_checkpoint('final.pt')
            print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train Text2Summary Model')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    # 设置分布式
    rank, local_rank, world_size = setup_distributed()

    try:
        # 加载配置
        if args.config and os.path.exists(args.config):
            config = load_config(args.config)
        else:
            config = create_default_config()

        # 创建训练器
        trainer = Trainer(config, rank, local_rank, world_size)

        # 恢复训练
        if args.resume:
            trainer.load_checkpoint(args.resume)

        # 开始训练
        trainer.train()

    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
