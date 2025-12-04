#!/bin/bash
# =============================================================================
# MeloFormer H800 Launch Script (v1.7.6 - 8k Stable)
# =============================================================================
# 诊断结论:
# - 16k: 73GB + 12GB = 85GB > 80GB → OOM
# - 12k: 78.5GB 接近极限，风险高
#
# 最终方案: 8k
# - 显存预计: ~55GB (安全余量 25GB)
# - 覆盖率: 86.6% (388,678 / 449,041 MIDI 文件)
# - 足够覆盖大多数流行歌曲
# =============================================================================

echo "=========================================="
echo "Starting MeloFormer Training (v1.7.6)"
echo "8k Stable Edition"
echo "=========================================="

# 1. 强力清理已有进程
echo "[1/4] Cleaning up existing processes..."
pkill -9 python 2>/dev/null || true
sleep 2

# 2. 清理 GPU 显存
echo "[2/4] Clearing GPU memory..."
nvidia-smi --gpu-reset 2>/dev/null || true

# 3. 显存优化环境变量 (防止碎片化)
echo "[3/4] Setting environment variables..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

# 4. 启动训练
echo "[4/4] Launching MeloFormer..."
echo ""
echo "Configuration:"
echo "  - Model: large (400M params)"
echo "  - Batch Size: 1 (effective: 16 with accumulation)"
echo "  - Max Seq Len: 8192 (8k stable)"
echo "  - Max Bars: 128"
echo "  - Static Compile: OFF"
echo "  - Precision: BF16"
echo "  - Gradient Checkpointing: ON (Sub-Layer GC)"
echo "  - Sequence Packing: ON"
echo ""
echo "Expected Memory: ~55GB (safe margin: 25GB)"
echo "Coverage: 86.6% of MIDI files"
echo ""

# v1.7.6: 8k 稳定版
nohup python -u train.py \
    --model_size large \
    --data_dir ~/autodl-tmp/arrow_data \
    --use_arrow \
    --batch_size 1 \
    --max_seq_len 8192 \
    --max_bars 128 \
    --gradient_accumulation_steps 16 \
    --epochs 20 \
    --learning_rate 1e-4 \
    --warmup_steps 2000 \
    --bf16 \
    --gradient_checkpointing \
    --use_packing \
    --num_workers 8 \
    --output_dir ./outputs_meloformer_large_12k \
    --log_interval 50 \
    --save_interval 2000 \
    > train_meloformer.log 2>&1 &

echo ""
echo "=========================================="
echo "MeloFormer 12k (v1.7.5) is running!"
echo "=========================================="
echo ""
echo "Monitor commands:"
echo "  - View logs:    tail -f train_meloformer.log"
echo "  - GPU status:   watch -n 1 nvidia-smi"
echo "  - Check process: ps aux | grep train.py"
echo ""
echo "12k = 12288 tokens ≈ 192 bars ≈ 完整的流行歌曲"
echo ""
