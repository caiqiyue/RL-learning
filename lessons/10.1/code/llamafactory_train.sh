#!/bin/bash
# LLaMA-Factory Training Launch Script
# Usage: bash llamafactory_train.sh

set -e

# ==================== Configuration ====================
# Model configuration
MODEL_NAME="LLaMA-3-8B-Instruct"
DATASET="my_instruct_data"
OUTPUT_DIR="./output/llama3-8b-lora"

# Training configuration
LEARNING_RATE="5.0e-5"
NUM_EPOCHS=3
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
LORA_RANK=8
LORA_ALPHA=16

# Quantization (set to 0 for no quantization, 4 or 8 for QLoRA)
QUANTIZATION_BIT=0

# ==================== Training Command ====================
CMD="llamafactory-cli train examples/train_full.yaml"

# Override with custom config
if [ -f "train_config.yaml" ]; then
    echo "Using custom train_config.yaml"
    CMD="llamafactory-cli train train_config.yaml"
fi

# ==================== Optional: Launch with DeepSpeed ====================
# DeepSpeed configuration (set to "" to disable)
# DEEPSPEED_CONFIG="examples/deepspeed/ds_config.json"

# ==================== Optional: Resume from checkpoint ====================
# RESUME_DIR="./output/llama3-8b-lora/checkpoint-1000"
# if [ -d "$RESUME_DIR" ]; then
#     CMD="$CMD --resume_from_checkpoint $RESUME_DIR"
# fi

# ==================== Execute Training ====================
echo "=========================================="
echo "LLaMA-Factory Training"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "=========================================="

eval $CMD

echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"