#!/bin/bash

# === 配置部分 ===
MODELS=(
    "ConvLSTM.py"
    "MAU.py"
    "MIM.py"
    "PhyDNet.py"
    "PredRNN.py"
    "PredRNNpp.py"
    "PredRNNv2.py"
    "SimVP_IncepU.py"
    "TAU.py"
)

CONFIG_DIR="/home/xingbo/method/OpenSTL/configs/weather/tcc_5_625"
LOG_DIR="/home/xingbo/method/OpenSTL/work_dirs/auto_logs"

mkdir -p $LOG_DIR


# === 每张卡一次只跑一个模型（按你要求修改） ===
run_group() {
    GPU=$1
    MODEL=$2

    MODEL_NAME=${MODEL%.py}

    echo "=== GPU$GPU 开始跑模型: $MODEL_NAME ==="

    CUDA_VISIBLE_DEVICES=$GPU python tools/train.py \
        --config_file $CONFIG_DIR/$MODEL \
        --dataname weather_tcc_5_625 \
        --data_root /scratch/xingbo \
        --ex_name $MODEL_NAME \
        > $LOG_DIR/${MODEL_NAME}.log 2>&1 &

    PID=$!

    echo "GPU$GPU 正在训练 PID: $PID"
    echo "等待 GPU$GPU 模型训练结束..."

    wait $PID

    echo "=== GPU$GPU 完成模型: $MODEL_NAME ==="
}


# === 分批次运行（结构与你原来完全一致，只是每组变成 1 个模型）===

# 批次 1
run_group 0 ${MODELS[0]} &
run_group 1 ${MODELS[1]}
wait

# 批次 2
run_group 0 ${MODELS[2]} &
run_group 1 ${MODELS[3]}
wait

# 批次 3
run_group 0 ${MODELS[4]} &
run_group 1 ${MODELS[5]}
wait

# 批次 4
run_group 0 ${MODELS[6]} &
run_group 1 ${MODELS[7]}
wait

# 批次 5（最后一个模型）
run_group 0 ${MODELS[8]} NONE

echo "===== 全部模型已训练完成 ====="
