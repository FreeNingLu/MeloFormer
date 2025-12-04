#!/bin/bash
# 解压并运行训练脚本

set -e  # 遇到错误立即退出

echo "========================================="
echo "MeloFormer v0.9.0 训练启动脚本"
echo "========================================="

# 解压代码
echo ""
echo "[1/3] 解压代码包..."
if [ -f "MeloFormer_v0.9.0.tar.gz" ]; then
    tar -xzf MeloFormer_v0.9.0.tar.gz
    echo "✅ 解压完成"
else
    echo "❌ 错误: 找不到 MeloFormer_v0.9.0.tar.gz"
    echo "请先上传压缩包到当前目录"
    exit 1
fi

# 检查数据集
echo ""
echo "[2/3] 检查数据集..."
if [ -d "data/preprocessed_shards" ]; then
    SHARD_COUNT=$(ls data/preprocessed_shards/shard_*.json 2>/dev/null | wc -l)
    echo "✅ 找到 $SHARD_COUNT 个数据分片"
else
    echo "❌ 警告: 未找到 data/preprocessed_shards/ 目录"
    echo "请确保数据集已上传"
fi

# 启动训练
echo ""
echo "[3/3] 启动训练..."
echo "========================================="

cd hid_museformer_v0.9

# 单卡训练命令
python3 train.py \
    --model_size small \
    --data_dir ../data/preprocessed_shards \
    --output_dir ../checkpoints \
    --max_seq_len 8192 \
    --batch_size 4 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --warmup_steps 1000 \
    --save_steps 5000 \
    --log_steps 10 \
    --use_amp \
    --autocast_dtype bfloat16

# 如果需要多卡 DDP 训练，使用下面的命令（替换上面的 python3 train.py）:
# torchrun --nproc_per_node=8 train.py \
#     --model_size small \
#     --data_dir ../data/preprocessed_shards \
#     --output_dir ../checkpoints \
#     --max_seq_len 8192 \
#     --batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --learning_rate 1e-4 \
#     --num_epochs 100 \
#     --warmup_steps 1000 \
#     --save_steps 5000 \
#     --log_steps 10 \
#     --use_amp \
#     --autocast_dtype bfloat16
