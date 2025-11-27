#!/bin/bash
NNODES=$1
NODE_RANK=$2
MASTER_ADDR=$3
MASTER_PORT=$4
DIST_CONFIG_PATH=$5
SAILOR_LOGS_DIR=$6

eval "mkdir -p $SAILOR_LOGS_DIR"

DP_SIZE=1
PP_SIZE=8
NODE_TP_SIZE=4
MAX_TP_SIZE=2

GLOBAL_BATCH_SIZE=128

MICRO_BATCH_SIZE=1
TRAIN_ITERS=5
MODEL_NAME=OPT

GA_STEPS=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DP_SIZE)))

config_json="ds_config.json"
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_accumulation_steps": $GA_STEPS,
  "zero_optimization": {},
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1.0e-5,
      "betas": [0.9, 0.999],
      "eps": 1.0e-8,
      "weight_decay": 4.0e-5
    }
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "consecutive_hysteresis": false,
    "min_loss_scale": 1
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

# MODEL ARGS
GPT_ARGS=" \
    --num-layers 32 \
    --num-transformer-layers-original 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
"

CONSTANT_ARGS=" \
    --loss-scale 12 \
    --lr 6.0e-5 \
    --min-lr 6.0e-6 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 40 \
    --eval-interval 1000 \
    --data-path /root/sailor/third_party/Megatron-DeepSpeed/data/meg-gpt2-oscar-en-10k_text_document \
    --vocab-file /root/sailor/third_party/Megatron-DeepSpeed/data/gpt2-vocab.json \
    --merge-file /root/sailor/third_party/Megatron-DeepSpeed/data/gpt2-merges.txt \
    --save-interval 1000 \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --fp16 \
"

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config $config_json \
"


LAUNCHER="
SAILOR_LOGS_DIR=$SAILOR_LOGS_DIR \
NCCL_IB_DISABLE=1 \
NCCL_IGNORE_DISABLED_P2P=1 \
NCCL_SOCKET_IFNAME=eth0 \
torchrun \
    --nproc_per_node $NODE_TP_SIZE \
    --nnodes=$NNODES \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
"

CMD=" \
    /root/sailor/third_party/Megatron-DeepSpeed/train_llm.py \
    --tensor-model-parallel-size $MAX_TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --data-parallel-size $DP_SIZE \
    --max-tensor-model-parallel-size $MAX_TP_SIZE \
    $GPT_ARGS \
    $CONSTANT_ARGS \
    $DEEPSPEED_ARGS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --train-iters $TRAIN_ITERS \
    --model-name $MODEL_NAME \
    --gpu-type RTX \
    --results-dir /root/sailor/third_party/Megatron-DeepSpeed/results \
    --distributed-config-file $DIST_CONFIG_PATH \
    --layers-per-stage 4 4 4 4 4 4 5 5 \
"

eval $LAUNCHER --node_rank $NODE_RANK $CMD