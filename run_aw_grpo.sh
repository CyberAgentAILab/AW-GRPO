#!/bin/bash

MODEL_NAME="sbintuitions/sarashina2.2-3b-instruct-v0.1"
TRAIN_DATASETS="dataset/wmt.en-ja/newstest2021.en-ja.all.xml,dataset/wmt.en-ja/wmttest2022.en-ja.all.xml,dataset/wmt.en-ja/wmttest2023.en-ja.all.xml"
TEST_DATASET="dataset/wmt.en-ja/wmttest2024.en-ja.all.xml"
OUTPUT_DIR="aw_grpo_results"
REWARD_FUNCTIONS="combined"
REWARD_WEIGHTS="readability=0.3,bleurt=0.3"
MAX_SEQ_LENGTH=512
LORA_RANK=128
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
NUM_GENERATIONS=8
LEARNING_RATE=2e-6
NUM_TRAIN_EPOCHS=3
SAVE_STEPS=150
TEMPERATURE=0.7
MAX_COMPLETION_LENGTH=200
VAL_SIZE=0.2
SEED=42
BLEURT_MODEL="lucadiliello/BLEURT-20-D12"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --train_datasets)
      TRAIN_DATASETS="$2"
      shift 2
      ;;
    --test_dataset)
      TEST_DATASET="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --reward_functions)
      REWARD_FUNCTIONS="$2"
      shift 2
      ;;
    --reward_weights)
      REWARD_WEIGHTS="$2"
      shift 2
      ;;
    --max_seq_length)
      MAX_SEQ_LENGTH="$2"
      shift 2
      ;;
    --lora_rank)
      LORA_RANK="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --gradient_accumulation_steps)
      GRADIENT_ACCUMULATION_STEPS="$2"
      shift 2
      ;;
    --num_generations)
      NUM_GENERATIONS="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --num_train_epochs)
      NUM_TRAIN_EPOCHS="$2"
      shift 2
      ;;
    --save_steps)
      SAVE_STEPS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --max_completion_length)
      MAX_COMPLETION_LENGTH="$2"
      shift 2
      ;;
    --val_size)
      VAL_SIZE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --bleurt_model)
      BLEURT_MODEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Print the parameters
echo "Running GRPO machine translation training with the following parameters:"
echo "  Model name: $MODEL_NAME"
echo "  Training datasets: $TRAIN_DATASETS"
echo "  Test dataset: $TEST_DATASET"
echo "  Output directory: $OUTPUT_DIR"
echo "  Reward functions: $REWARD_FUNCTIONS"
echo "  Reward weights: $REWARD_WEIGHTS"
echo "  Max sequence length: $MAX_SEQ_LENGTH"
echo "  LoRA rank: $LORA_RANK"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  Number of generations: $NUM_GENERATIONS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Number of training epochs: $NUM_TRAIN_EPOCHS"
echo "  Save steps: $SAVE_STEPS"
echo "  Temperature: $TEMPERATURE"
echo "  Max completion length: $MAX_COMPLETION_LENGTH"
echo "  Validation size: $VAL_SIZE"
echo "  Random seed: $SEED"
echo "  BLEURT model: $BLEURT_MODEL"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Start resource logger in the background
echo "Starting resource logger..."
python3 resource_logger.py --log_file "${OUTPUT_DIR}/resource_log.txt" --interval 1 --disk_paths / /pg --format csv --quiet &
LOGGER_PID=$!

# Run the Python script
echo "Starting main training script..."
python3 aw_grpo.py \
  --model_name "$MODEL_NAME" \
  --train_datasets "$TRAIN_DATASETS" \
  --test_dataset "$TEST_DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --reward_functions "$REWARD_FUNCTIONS" \
  --reward_weights "$REWARD_WEIGHTS" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --lora_rank "$LORA_RANK" \
  --batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --num_generations "$NUM_GENERATIONS" \
  --learning_rate "$LEARNING_RATE" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --save_steps "$SAVE_STEPS" \
  --temperature "$TEMPERATURE" \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --val_size "$VAL_SIZE" \
  --seed "$SEED" \
  --bleurt_model "$BLEURT_MODEL" \

# Stop the resource logger
echo "Stopping resource logger..."
kill $LOGGER_PID

# Print completion message
echo "GRPO machine translation training completed."