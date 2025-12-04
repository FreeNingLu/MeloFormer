# ⚡ H800 优化训练版 - FIXED VERSION
"""
🚀 H800 优化配置（80GB显存）：

硬件规格：
- GPU: H800 80GB (Hopper架构)
- 驱动: 570.124.04
- CUDA: 12.8
- CPU: 20核 Xeon Platinum 8458P
- 内存: 100GB

🔧 关键修复（基于GPT建议）：

1. ✅ 修复 Padding Mask Bug（最重要）
   - 问题：padding的0被当成正常token参与loss计算
   - 修复：CrossEntropyLoss设置ignore_index=0
   - 预期：loss下降更快，生成质量提升

2. ✅ 关闭 Label Smoothing
   - 问题：0.1的smoothing在30k vocab下过大，导致欠拟合
   - 修复：label_smoothing从0.1改为0
   - 预期：模型更容易学到确定性预测

3. ✅ 增加训练量
   - 问题：10 epochs太少，模型欠拟合
   - 修复：epochs从10提高到20，learning_rate从5e-6提高到1e-5
   - 预期：模型充分学习MIDI时序结构

🚀 H800 性能优化（相比RTX 5090）：

1. ✅ 增大batch size（2→8，充分利用80GB显存）
2. ✅ 减少gradient accumulation（32→8，保持有效batch=64）
3. ✅ 增大最大序列长度（15000→15000，处理更长MIDI）
4. ✅ 增加数据加载workers（8→16，20核CPU）
5. ✅ Flash Attention 2（Hopper架构原生优化）
6. ✅ 使用BF16混合精度训练
7. ✅ 提高学习率（5e-6→1e-5）

预计效果：
- 训练时间：约3-4天（20 epochs，比RTX 5090快50%+）
- Loss下降：更快、更稳定
- 生成质量：显著提升
- 支持更长MIDI序列（15000 tokens vs 15000）


python train_FAST_H800_fixed.py \
    --caption_path /root/autodl-tmp/tmp/captions/captionMidiCapsPlus.json \
    --midi_folder /root/autodl-tmp/tmp/MidiCaps \
    --output_dir output_H800 \
    --epochs 20

    python train_FAST_H800_fixed.py --caption_path /root/autodl-tmp/Text2midi/captions/captionMidiCapsPlus.json --midi_folder /root/autodl-tmp/Text2midi/MidiCaps --output_dir /root/autodl-tmp/Text2midi/output_MidiCaps_H800_FIXED > training.log 2>&1 &

"""

import os
import torch.nn as nn
import torch.optim as optim
import yaml
import math
import time
from transformers import get_scheduler
import wandb
import pickle
import numpy as np
import json
import jsonlines
from tqdm import tqdm
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
import logging
import argparse

# 导入模型和数据加载器
import sys
sys.path.append('/root/autodl-tmp/Text2midi/model')
from data_loader_remi import Text2MusicDataset

# ✅ 导入模型（H800同样支持Flash Attention 2）
from model.transformer_model_RoPE_Flash2_KVCache_RTX5090 import Transformer, FLASH_ATTN_AVAILABLE

from torch.utils.data import DataLoader

logger = get_logger(__name__)

# ==================== 全局常量 ====================
PAD_ID = 0  # REMI tokenizer的padding ID


# ==================== 配置参数 ====================
class H800FixedTrainingConfig:
    """修复版训练配置 - H800 80GB优化"""
    def __init__(self):
        # 路径配置
        self.config_file = "configs/config.yaml"
        self.pretrained_model_path = None
        self.vocab_path = "artifacts/vocab_remi.pkl"

        self.caption_dataset_path = "/root/autodl-tmp/Text2midi/captions/captionMidiCapsPlus.json"
        self.midi_folder_path = r"/root/autodl-tmp/Text2midi/MidiCaps"
        self.output_dir = "/root/autodl-tmp/Text2midi/output_MidiCaps_H800_FIXED"

        # 模型配置
        self.decoder_d_model = 768
        self.decoder_num_heads = 12
        self.decoder_num_layers = 18

        # 🚀 H800优化：增大最大序列长度（80GB显存）
        self.decoder_max_sequence_length = 15000  # ⬆️ 从15000提高到15000
        self.filter_long_samples = True

        self.decoder_intermediate_size = 3072
        self.use_moe = False
        self.num_experts = 4

        # 🔧 修复2：增加训练量，提高学习率
        self.epochs = 30  # ⬆️ 从10提高到20
        self.learning_rate = 1e-5  # ⬆️ 从5e-6提高到1e-5

        # 🚀 H800优化：增大batch size，减少gradient accumulation（梯度检查点）
        self.per_device_train_batch_size = 6  # 启用梯度检查点后可用6
        self.gradient_accumulation_steps = 11  # 保持有效batch=66

        # 🚀 H800优化：调整warmup和保存频率
        self.num_warmup_steps = 15000
        self.lr_scheduler_type = "cosine"
        self.save_every = 5  # 每2个epoch保存

        self.max_train_steps = None

        # KV Cache配置
        self.max_cache_len = 15000  # ⬆️ 匹配max_sequence_length

        # ⚡ 优化5：启用梯度检查点（节省显存30-40%）
        self.use_gradient_checkpointing = True  # 允许更大batch size

        # ⚡ 优化6：启用torch.compile加速
        self.use_torch_compile = False  # PyTorch 2.7新特性

        # 实验追踪配置
        self.with_tracking = True
        self.report_to = "wandb"
        self.project_name = "MidiCaps-FIXED-H800-80GB"

        # 加载yaml配置
        self.load_yaml_config()

    def load_yaml_config(self):
        try:
            with open(self.config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
                self.artifact_folder = yaml_config.get('artifact_folder', 'artifacts')
        except Exception as e:
            print(f"Warning: Could not load yaml config: {e}")
            self.artifact_folder = 'artifacts'


def collate_fn(batch):
    """
    数据批处理函数

    注意：padding使用PAD_ID=0，但会在loss计算时被ignore
    """
    input_ids = [item[0].squeeze(0) for item in batch]
    attention_mask = [item[1].squeeze(0) for item in batch]
    labels = [item[2] if item[2].dim() == 1 else item[2].squeeze(0) for item in batch]

    # Padding with PAD_ID=0
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=PAD_ID)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PAD_ID)

    return input_ids, attention_mask, labels


def setup_training(config):
    """设置训练环境"""
    accelerator_log_kwargs = {}
    if config.with_tracking:
        accelerator_log_kwargs["log_with"] = config.report_to
        accelerator_log_kwargs["project_dir"] = config.output_dir

    # ✅ H800: BF16混合精度（Hopper架构原生支持）
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision='bf16',
        **accelerator_log_kwargs
    )

    # 配置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        logger.info("="*70)
        logger.info("🚀 FIXED Training Configuration - H800 80GB")
        logger.info("="*70)
        logger.info(f"GPU: H800 (80GB) x1 - Hopper架构")
        logger.info(f"Max Seq Len: {config.decoder_max_sequence_length} (⬆️ vs RTX5090: 15000)")
        logger.info(f"Batch Size: {config.per_device_train_batch_size} (⬆️ vs RTX5090: 2)")
        logger.info(f"Gradient Acc: {config.gradient_accumulation_steps} (⬇️ vs RTX5090: 32)")
        logger.info(f"Effective Batch: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
        logger.info(f"")
        logger.info(f"🔧 KEY FIXES:")
        logger.info(f"  1. Padding Mask: ignore_index={PAD_ID} ✅")
        logger.info(f"  2. Label Smoothing: 0 (was 0.1) ✅")
        logger.info(f"  3. Epochs: {config.epochs} (was 10) ✅")
        logger.info(f"  4. Learning Rate: {config.learning_rate} (was 5e-6) ✅")
        logger.info(f"")
        logger.info(f"🚀 H800 OPTIMIZATIONS:")
        logger.info(f"  1. Batch Size: 2→8 (4x larger)")
        logger.info(f"  2. Gradient Acc: 32→8 (4x less)")
        logger.info(f"  3. Max Seq Len: 15000→15000 (33% longer)")
        logger.info(f"  4. DataLoader workers: 8→16 (20核CPU)")
        logger.info(f"")

        if FLASH_ATTN_AVAILABLE:
            logger.info("✅ Flash Attention 2 ENABLED (Hopper优化)")
        else:
            logger.warning("⚠️ Flash Attention NOT available")

        logger.info("="*70)

    # 创建输出目录
    if accelerator.is_main_process:
        if config.output_dir is None or config.output_dir == "":
            config.output_dir = "saved/" + str(int(time.time()))
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)

        # 初始化wandb
        if config.with_tracking:
            wandb.login()
            wandb.init(
                project=config.project_name,
                name=f"fixed-h800-{time.strftime('%Y%m%d-%H%M%S')}",
                settings=wandb.Settings(init_timeout=120),
                config={
                    "gpu": "H800 (80GB) Hopper",
                    "learning_rate": config.learning_rate,
                    "epochs": config.epochs,
                    "batch_size": config.per_device_train_batch_size,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "max_seq_len": config.decoder_max_sequence_length,
                    "optimization": "H800 + FIXED + Flash Attention 2 + BF16",
                    "fixes": "padding_mask + label_smoothing + training_amount",
                    "label_smoothing": 0,
                    "ignore_padding": True,
                    "h800_optimizations": "8x_batch + 20k_seq + 16workers"
                }
            )

    accelerator.wait_for_everyone()
    return accelerator


def load_dataset(config, tokenizer, accelerator):
    """🚀 H800优化的数据加载（16 workers，20核CPU）"""
    logger.info("Loading dataset...")

    with jsonlines.open(config.caption_dataset_path) as reader:
        captions = list(reader)

    logger.info(f"Loaded {len(captions)} captions")

    # ⚡ 过滤超长样本
    if config.filter_long_samples:
        logger.info(f"Filter enabled for samples > {config.decoder_max_sequence_length} tokens")

    temp_config = {
        'raw_data': {
            'raw_data_folders': {
                'MidiCaps': {
                    'folder_path': config.midi_folder_path,
                    'file_extension': 'mid'
                }
            }
        },
        'model': {
            'text2midi_model': {
                'decoder_max_sequence_length': config.decoder_max_sequence_length
            }
        },
        'artifact_folder': config.artifact_folder
    }

    with accelerator.main_process_first():
        dataset = Text2MusicDataset(
            temp_config,
            captions,
            remi_tokenizer=tokenizer,
            mode="train",
            shuffle=True
        )

        # 🚀 H800优化：增加workers到16（20核CPU）
        dataloader = DataLoader(
            dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True,
            num_workers=8,  # ⬆️ 从8提高到16
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=8,#4
            persistent_workers=True
        )

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"DataLoader: workers=16, prefetch=4 (H800优化)")
    return dataset, dataloader


def create_or_load_model(config, vocab_size, device):
    """创建或加载模型"""
    logger.info("Creating model...")

    model = Transformer(
        n_vocab=vocab_size,
        d_model=config.decoder_d_model,
        nhead=config.decoder_num_heads,
        max_len=config.decoder_max_sequence_length,
        num_decoder_layers=config.decoder_num_layers,
        dim_feedforward=config.decoder_intermediate_size,
        use_moe=config.use_moe,
        num_experts=config.num_experts,
        max_cache_len=config.max_cache_len,
        device=device
    )

    # 加载预训练权重
    if config.pretrained_model_path and os.path.exists(config.pretrained_model_path):
        logger.info(f"Loading pretrained model: {config.pretrained_model_path}")
        try:
            state_dict = torch.load(config.pretrained_model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded pretrained model!")
        except Exception as e:
            logger.warning(f"Failed to load: {e}")
    else:
        logger.info("Training from scratch")

    # 🔧 关键修复：启用梯度检查点来节省显存
    if config.use_gradient_checkpointing:
        logger.info("="*70)
        logger.info("🔧 启用梯度检查点 (Gradient Checkpointing)")
        logger.info("="*70)
        logger.info("预期效果：")
        logger.info("  - 显存节省: ~30-40% ✅")
        logger.info("  - 速度影响: ~10-15%慢 (可接受)")
        logger.info("  - 允许batch size: 6-8 (vs 当前4)")
        logger.info("="*70)

        # 启用梯度检查点：修改decoder的forward方法来使用checkpointing
        model.decoder.gradient_checkpointing = True
        logger.info(f"✅ 已为 {len(model.decoder.layers)} 个decoder层启用梯度检查点")

    # ⚡ 优化：使用torch.compile加速
    if config.use_torch_compile:
        try:
            logger.info("⚡ Compiling model with torch.compile...")
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("✅ Model compiled successfully!")
        except Exception as e:
            logger.warning(f"⚠️ torch.compile failed: {e}")
            logger.warning("Continuing without compilation...")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params:,}")

    return model


def train_model(config, model, dataloader, accelerator, device):
    """🔧 修复版训练循环 - H800优化"""
    model.train()
    # ===== 添加这段验证代码 =====
    print("\n" + "="*70)
    print("🔍 验证Flash Attention状态")
    print("="*70)
    print(f"模型training模式: {model.training}")
    print(f"Flash Attention可用:{model.decoder.layers[0].self_attn.forward.__code__.co_filename}")
    # 测试一次forward看走哪个分支
    test_input = torch.randn(2, 100, 768, device=device)
    layer = model.decoder.layers[0].self_attn
    print(f"\n测试attention forward:")
    print(f"  - layer.training = {layer.training}")
    print(f"  - 应该使用 Flash Attention: {layer.training}")
    # ===== 验证代码结束 =====
    
    logger.info("Starting FIXED training on H800...")

    # 设置优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.05
    )

    # 计算训练步数
    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / config.gradient_accumulation_steps
    )

    if config.max_train_steps is None:
        max_train_steps = config.epochs * num_update_steps_per_epoch
    else:
        max_train_steps = config.max_train_steps
        config.epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # 学习率调度器
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(0.10 * max_train_steps),  # 10% warmup
        num_training_steps=max_train_steps,
    )

    # Prepare with accelerator
    model, optimizer, lr_scheduler, dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, dataloader
    )

    # 训练信息
    total_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps

    logger.info("***** H800 FIXED Training Configuration *****")
    logger.info(f"  Num Epochs = {config.epochs}")
    logger.info(f"  Batch size per device = {config.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation = {config.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Max Seq Length = {config.decoder_max_sequence_length}")
    logger.info(f"  Learning Rate = {config.learning_rate}")
    logger.info(f"  Label Smoothing = 0 (FIXED)")
    logger.info(f"  Ignore Padding = True (FIXED)")

    # 🚀 H800预估训练时间（梯度检查点 + batch=6）
    # 梯度检查点增加15% overhead，但减少累积步数补偿
    estimated_time_per_step = 11  # batch=6, grad_acc=11, 约11秒/步
    total_hours = (max_train_steps * estimated_time_per_step) / 3600
    logger.info(f"  Estimated training time: {total_hours:.1f} hours ({total_hours/24:.1f} days)")
    logger.info(f"  Estimated time/step: {estimated_time_per_step}s (with gradient checkpointing)")
    logger.info(f"  (H800: ~{total_hours/24:.1f} days vs RTX5090: ~{total_hours/24*1.5:.1f} days)")

    # 🔧 修复1: 损失函数 - 添加ignore_index=PAD_ID
    # 🔧 修复2: Label Smoothing设为0
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0,  # 🔧 从0.1改为0
        ignore_index=PAD_ID  # 🔧 关键修复：忽略padding的0
    )

    logger.info(f"✅ Loss function configured:")
    logger.info(f"   - ignore_index={PAD_ID} (padding will not contribute to loss)")
    logger.info(f"   - label_smoothing=0 (disabled for better convergence)")

    # 训练循环
    progress_bar = tqdm(
        range(max_train_steps),
        disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0
    best_loss = float('inf')

    model.train()

    # 显存监控
    if accelerator.is_main_process and torch.cuda.is_available():
        logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / 80 GB")

    # ⚡ 训练开始时间
    training_start_time = time.time()
    step_times = []

    for epoch in range(starting_epoch, config.epochs):
        total_loss = 0
        epoch_loss = 0
        epoch_start_time = time.time()

        for step, batch in enumerate(dataloader):
            step_start_time = time.time()

            with accelerator.accumulate(model):
                encoder_input, attention_mask, tgt = batch

                encoder_input = encoder_input.to(device)
                attention_mask = attention_mask.to(device)
                tgt = tgt.to(device)

                # 准备decoder输入和目标
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                # 前向传播
                if config.use_moe:
                    outputs, aux_loss = model(encoder_input, attention_mask, tgt_input)
                else:
                    outputs = model(encoder_input, attention_mask, tgt_input)
                    aux_loss = 0

                # 🔧 计算损失（padding已被自动忽略）
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)),
                    tgt_output.reshape(-1)
                )
                loss += aux_loss

                total_loss += loss.detach().float()
                epoch_loss += loss.detach().float()

                # 反向传播
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 更新进度条
            if accelerator.sync_gradients:
                step_time = time.time() - step_start_time
                step_times.append(step_time)

                # 计算平均步时间和剩余时间
                if len(step_times) > 10:
                    avg_step_time = np.mean(step_times[-100:])
                    remaining_steps = max_train_steps - completed_steps
                    eta_seconds = remaining_steps * avg_step_time
                    eta_hours = eta_seconds / 3600

                    progress_bar.set_postfix({
                        "epoch": epoch + 1,
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        "step_time": f"{step_time:.1f}s",
                        "eta": f"{eta_hours:.1f}h"
                    })
                else:
                    progress_bar.set_postfix({
                        "epoch": epoch + 1,
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        "step_time": f"{step_time:.1f}s"
                    })

                progress_bar.update(1)
                completed_steps += 1

                # 记录到wandb
                if accelerator.is_main_process and config.with_tracking:
                    log_dict = {
                        "train_loss": loss.item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch + 1,
                        "step": completed_steps,
                        "step_time": step_time,
                        "sequence_length": tgt_input.size(1)
                    }

                    if torch.cuda.is_available():
                        log_dict["gpu_memory_gb"] = torch.cuda.memory_allocated()/1e9

                    wandb.log(log_dict)

            if completed_steps >= max_train_steps:
                break

        # Epoch结束
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss.item() / len(dataloader)

        if accelerator.is_main_process:
            logger.info(f"\nEpoch {epoch+1}/{config.epochs} completed in {epoch_time/60:.1f} min")
            logger.info(f"Avg Loss: {avg_epoch_loss:.4f}")

            # 显存统计
            if torch.cuda.is_available():
                current_mem = torch.cuda.memory_allocated()/1e9
                peak_mem = torch.cuda.max_memory_allocated()/1e9
                logger.info(f"GPU Memory: {current_mem:.2f} GB / {peak_mem:.2f} GB (peak) / 80 GB total")
                torch.cuda.reset_peak_memory_stats()

            # 保存checkpoint
            if (epoch + 1) % config.save_every == 0:
                checkpoint_dir = f"{config.output_dir}/checkpoints/epoch_{epoch+1}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(
                    unwrapped_model.state_dict(),
                    f"{checkpoint_dir}/pytorch_model.bin"
                )
                logger.info(f"Checkpoint saved to {checkpoint_dir}")

            # 保存最佳模型
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_dir = f"{config.output_dir}/best_model"
                os.makedirs(best_model_dir, exist_ok=True)

                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(
                    unwrapped_model.state_dict(),
                    f"{best_model_dir}/pytorch_model.bin"
                )
                logger.info(f"Best model saved (loss: {best_loss:.4f})")

        accelerator.wait_for_everyone()

    # 训练完成
    total_training_time = time.time() - training_start_time

    if accelerator.is_main_process:
        logger.info("\n" + "="*70)
        logger.info("🚀 H800 FIXED Training Completed!")
        logger.info(f"Total time: {total_training_time/3600:.2f} hours ({total_training_time/86400:.2f} days)")
        logger.info(f"Best loss: {best_loss:.4f}")
        logger.info("="*70)

        final_model_dir = f"{config.output_dir}/final_model"
        os.makedirs(final_model_dir, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(
            unwrapped_model.state_dict(),
            f"{final_model_dir}/pytorch_model.bin"
        )
        logger.info(f"Final model saved to {final_model_dir}")

        if config.with_tracking:
            wandb.finish()


def main():
    """主函数 - H800 修复版"""
    os.environ['WANDB_MODE'] = 'offline'

    parser = argparse.ArgumentParser(
        description="🚀 H800 FIXED Training with Flash Attention 2 (80GB)"
    )
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--caption_path", type=str)
    parser.add_argument("--midi_folder", type=str)
    parser.add_argument("--output_dir", type=str, default="output_H800_FIXED")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=6)  # H800 with gradient checkpointing
    parser.add_argument("--gradient_accumulation", type=int, default=11)  # 保持有效batch=66
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_seq_len", type=int, default=15000)  # H800默认15000
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True)  # 默认启用
    args = parser.parse_args()

    # 创建配置
    config = H800FixedTrainingConfig()

    # 从命令行参数更新配置
    if args.pretrained_model:
        config.pretrained_model_path = args.pretrained_model
    config.caption_dataset_path = args.caption_path
    print("\n" + "="*70)
    print(f"caption_dataset_path: {config.caption_dataset_path}")
    config.midi_folder_path = args.midi_folder
    print(f"midi_folder_path: {config.midi_folder_path}")
    config.output_dir = args.output_dir
    print(f"output_dir: {config.output_dir}")
    config.epochs = args.epochs
    config.per_device_train_batch_size = args.batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation
    config.learning_rate = args.learning_rate
    config.decoder_max_sequence_length = args.max_seq_len
    config.use_gradient_checkpointing = args.use_gradient_checkpointing

    print("\n" + "="*70)
    print("🚀 H800 FIXED Training Configuration (80GB)")
    print("="*70)
    print(f"\n🔧 KEY FIXES:")
    print(f"  1. Padding Mask: ignore_index={PAD_ID} ✅")
    print(f"  2. Label Smoothing: 0 (was 0.1) ✅")
    print(f"  3. Epochs: {config.epochs} (was 10) ✅")
    print(f"  4. Learning Rate: {config.learning_rate} (was 5e-6) ✅")
    print(f"  5. Gradient Checkpointing: {'启用' if config.use_gradient_checkpointing else '禁用'} ✅")
    print(f"\n🚀 H800 OPTIMIZATIONS:")
    print(f"  Batch Size: {config.per_device_train_batch_size} (vs RTX5090: 2)")
    print(f"  Gradient Acc: {config.gradient_accumulation_steps} (vs RTX5090: 32)")
    print(f"  Effective Batch: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"  Max Seq Len: {config.decoder_max_sequence_length} (vs RTX5090: 15000)")
    print(f"  DataLoader: workers=8, prefetch=8 (vs RTX5090: 8workers)")
    print(f"  梯度检查点: {'启用(节省30-40%显存)' if config.use_gradient_checkpointing else '禁用'}")
    print("="*70)

    # 计算估算
    total_steps = (168385 / config.per_device_train_batch_size) * config.epochs / config.gradient_accumulation_steps

    # 梯度检查点会增加~15%计算时间，但减少梯度累积步数可以抵消
    # batch=6, grad_acc=11: 每个step需要11次forward/backward
    # 预估: 0.8s GPU计算 + 11×0.9s 累积 = ~10.7s/step (考虑梯度检查点overhead)
    estimated_time_per_step = 11  # 秒
    estimated_hours = (total_steps * estimated_time_per_step) / 3600
    estimated_days = estimated_hours / 24

    rtx5090_steps = (168385 / 2) * 20 / 32
    rtx5090_hours = (rtx5090_steps * 15) / 3600
    rtx5090_days = rtx5090_hours / 24

    print(f"\n⏱️ Estimated Training Time:")
    print(f"  Total steps: {total_steps:,.0f}")
    print(f"  Estimated time/step: {estimated_time_per_step}s (with gradient checkpointing)")
    print(f"  H800: ~{estimated_days:.1f} days ({estimated_hours:.0f} hours)")
    print(f"  RTX5090 (对比): ~{rtx5090_days:.1f} days ({rtx5090_hours:.0f} hours)")
    print(f"  Speed-up: {rtx5090_days/estimated_days:.1f}x faster ⚡")
    print("="*70 + "\n")

    # 设置训练环境
    accelerator = setup_training(config)
    device = accelerator.device

    # 加载tokenizer
    logger.info(f"Loading tokenizer from: {config.vocab_path}")
    with open(config.vocab_path, "rb") as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer)
    logger.info(f"Vocab size: {vocab_size}")

    # 加载数据集
    dataset, dataloader = load_dataset(config, tokenizer, accelerator)

    # 创建或加载模型
    model = create_or_load_model(config, vocab_size, device)

    # 开始训练
    train_model(config, model, dataloader, accelerator, device)


if __name__ == "__main__":
    main()
