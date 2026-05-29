#!/bin/bash

set -e

NUM_GPUS=${1:-8}
MODEL_NAME=${2:-"facebook/opt-1.3b"}
OUTPUT_DIR=${3:-"./output"}
DEEPSPEED_CONFIG=${4:-"ds_config.json"}

if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "Error: DeepSpeed config file not found: $DEEPSPEED_CONFIG"
    exit 1
fi

echo "=========================================="
echo "DeepSpeed Multi-GPU Training Launcher"
echo "=========================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "DeepSpeed config: $DEEPSPEED_CONFIG"
echo "=========================================="

NUM_GPUS_AVAILABLE=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

if [ "$NUM_GPUS" -gt "$NUM_GPUS_AVAILABLE" ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $NUM_GPUS_AVAILABLE available"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

GPUS_SELECTED=""
for ((i=0; i<NUM_GPUS; i++)); do
    GPUS_SELECTED="${GPUS_SELECTED}${GPUS_SELECTED:+,}${i}"
done

echo "Selected GPUs: $GPUS_SELECTED"

deepspeed --num_gpus=$NUM_GPUS \
    --master_port=29500 \
    train_deepspeed.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name "wikitext" \
    --dataset_config "wikitext-2-raw-v1" \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_length 512 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --deepspeed "$DEEPSPEED_CONFIG"

echo ""
echo "=========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="