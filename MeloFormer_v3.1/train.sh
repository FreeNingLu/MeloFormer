#!/usr/bin/env bash
# MeloFormer v2.2 训练启动脚本 (H800)
#
# 两种训练模式 (统一使用 train.py):
#
#   pretrain  Summary Token + Phase 1/2 双阶段注意力自回归预训练
#             train_obj: 下一个 token 预测 (无 FIM 重排)
#
#   fim       在预训练权重基础上做 FIM 微调 (--fim 标志)
#             支持 4 种模式: PRETRAIN / TRACK_MASK / TIME_FIM / SKELETON_FIM
#
# 用法:
#   ./train.sh --data-dir /path/to/arrow                        # pretrain (默认)
#   ./train.sh --data-dir /path/to/arrow --mode fim \           # FIM 微调
#              --pretrained runs/xxx/checkpoints/best.pt
#   ./train.sh --data-dir /path/to/arrow --resume runs/xxx/checkpoints/step_5000.pt
#   ./train.sh --data-dir /path/to/arrow --model 177m --lr 3e-5 --epochs 10
#   ./train.sh --data-dir /path/to/arrow --mode fim --fim-ratio 0.8
#
# 监控:
#   screen -r meloformer_pretrain        # pretrain 模式
#   screen -r meloformer_fim             # fim 模式
#   tail -f runs/<run_name>/logs/stdout.log
#   tail -f runs/<run_name>/logs/train_*.log

set -euo pipefail

NVIDIA_LIBS=/root/highspeedstorage/plume/pyenv/lib/python3.11/site-packages/nvidia
export TORCHINDUCTOR_CACHE_DIR=/root/highspeedstorage/plume/pyenv/torch_inductor_cache
export TRITON_CACHE_DIR=/root/highspeedstorage/plume/pyenv/triton_cache
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:${NVIDIA_LIBS}/cudnn/lib:${NVIDIA_LIBS}/nccl/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD="${NVIDIA_LIBS}/nccl/lib/libnccl.so.2 /usr/local/cuda-12.9/compat/libcuda.so.575.51.03 /usr/local/cuda-12.9/compat/libnvidia-ptxjitcompiler.so.575.51.03"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR=""          # 必须通过 --data-dir 指定
RUNS_DIR="${SCRIPT_DIR}/runs"

# ---------------------------------------------------------------------------
# 默认超参
# ---------------------------------------------------------------------------
MODE="pretrain"        # pretrain | fim
MODEL_SIZE="1.5b"
EPOCHS="1"
LR="1e-4"
WARMUP="2000"
GRAD_ACCUM="64"
MAX_SEQ_LEN="16384"
MAX_CHORDS="256"
NUM_WORKERS="16"
LOG_INTERVAL="5"
SAVE_INTERVAL="1000"
RESUME_ARG=""
PRETRAINED_ARG=""
FIM_RATIO="0.7"
MODE_WEIGHTS="0.3,0.25,0.25,0.2"  # PRETRAIN,TRACK_MASK,SKELETON_FIM,TIME_FIM
RUN_NAME=""

# v3.0 架构优化开关 (默认关闭)
SUB_ATTN_BIAS=""
K2V2_GATE=""
DEPTH_SELECTOR=""

# ---------------------------------------------------------------------------
# 解析参数
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)         MODE="$2";                           shift 2 ;;
        --data-dir)     DATA_DIR="$2";                       shift 2 ;;
        --resume)       RESUME_ARG="--resume $2";            shift 2 ;;
        --pretrained)   PRETRAINED_ARG="--pretrained $2";    shift 2 ;;
        --lr)           LR="$2";                             shift 2 ;;
        --epochs)       EPOCHS="$2";                         shift 2 ;;
        --model)        MODEL_SIZE="$2";                     shift 2 ;;
        --run-name)     RUN_NAME="$2";                       shift 2 ;;
        --fim-ratio)    FIM_RATIO="$2";                      shift 2 ;;
        --mode-weights) MODE_WEIGHTS="$2";                   shift 2 ;;
        --grad-accum)   GRAD_ACCUM="$2";                     shift 2 ;;
        --max-seq-len)  MAX_SEQ_LEN="$2";                    shift 2 ;;
        --sub-attn-bias)    SUB_ATTN_BIAS="--sub_attn_bias";    shift 1 ;;
        --k2v2-gate)        K2V2_GATE="--k2v2_gate";            shift 1 ;;
        --depth-selector)   DEPTH_SELECTOR="--depth_selector";  shift 1 ;;
        *) echo "[!] 未知参数: $1"; exit 1 ;;
    esac
done

# 校验
if [[ -z "${DATA_DIR}" ]]; then
    echo "[!] 必须通过 --data-dir 指定数据集路径"
    exit 1
fi
if [[ "${MODE}" != "pretrain" && "${MODE}" != "fim" ]]; then
    echo "[!] --mode 必须是 pretrain 或 fim"
    exit 1
fi
if [[ "${MODE}" == "fim" && -z "${PRETRAINED_ARG}" && -z "${RESUME_ARG}" ]]; then
    echo "[!] FIM 模式建议通过 --pretrained 指定预训练权重路径"
    echo "    继续? (y/N)" && read -r ans && [[ "${ans}" =~ ^[Yy]$ ]] || exit 1
fi

# 默认 run 名称（在解析完 MODE 后设置）
if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="meloformer_${MODEL_SIZE}_${MODE}_$(date +%Y%m%d_%H%M%S)"
fi

# screen session 名称
SESSION="meloformer_${MODE}"

OUTPUT_DIR="${RUNS_DIR}/${RUN_NAME}"
LOG_FILE="${OUTPUT_DIR}/logs/stdout.log"

# ---------------------------------------------------------------------------
# 准备目录
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}/logs"

# ---------------------------------------------------------------------------
# 停止已有同名 screen session
# ---------------------------------------------------------------------------
if screen -list 2>/dev/null | grep -q "${SESSION}"; then
    echo "[!] 已存在 screen session '${SESSION}'，先停止..."
    screen -S "${SESSION}" -X quit 2>/dev/null || true
    sleep 1
fi

# ---------------------------------------------------------------------------
# 构建训练命令
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "fim" ]]; then
    SCRIPT="train.py"
    EXTRA_ARGS="--fim --fim_ratio ${FIM_RATIO} --mode_weights ${MODE_WEIGHTS} --use_packing ${PRETRAINED_ARG}"
else
    SCRIPT="train.py"
    EXTRA_ARGS="--batch_size 1 --use_packing"
fi

TRAIN_CMD="cd ${SCRIPT_DIR} && .venv/bin/python -u ${SCRIPT} \
    --data_dir ${DATA_DIR} \
    --use_arrow \
    --model_size ${MODEL_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --bf16 \
    --gradient_checkpointing \
    --max_seq_len ${MAX_SEQ_LEN} \
    --max_chords ${MAX_CHORDS} \
    --learning_rate ${LR} \
    --warmup_steps ${WARMUP} \
    --epochs ${EPOCHS} \
    --output_dir ${OUTPUT_DIR} \
    --num_workers ${NUM_WORKERS} \
    --log_interval ${LOG_INTERVAL} \
    --save_interval ${SAVE_INTERVAL} \
    ${EXTRA_ARGS} \
    ${SUB_ATTN_BIAS} ${K2V2_GATE} ${DEPTH_SELECTOR} \
    ${RESUME_ARG} \
    2>&1 | tee -a ${LOG_FILE}"

# ---------------------------------------------------------------------------
# 用 screen -dm 后台启动，detach 不影响训练进程
# ---------------------------------------------------------------------------
screen -dmS "${SESSION}" bash -c "${TRAIN_CMD}; exec bash"

echo ""
echo "========================================"
echo "  MeloFormer v3.1 训练已启动 [${MODE}]"
echo "========================================"
echo "  Mode      : ${MODE}"
echo "  Script    : ${SCRIPT}"
echo "  Backend   : TRITON (dynamic=True)"
echo "  Run name  : ${RUN_NAME}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Log file  : ${LOG_FILE}"
echo "  screen 会话: ${SESSION}"
echo ""
echo "  切回窗口 : screen -r ${SESSION}"
echo "  查看日志 : tail -f ${LOG_FILE}"
echo "  停止训练 : pkill -f ${SCRIPT}"
echo "========================================"
