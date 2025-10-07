import argparse
import os
import json
import logging
from datetime import datetime
from typing import List
import torch
from tqdm import tqdm
import numpy as np
from datasets import Dataset
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig, GRPOTrainer
from reward_functions import RewardRegistry, parse_reward_weights
from dataset_utils import load_datasets, prepare_dataset_for_grpo, split_dataset, create_translation_prompt

logger = logging.getLogger(__name__)
def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Path to log file (default: None, logs to console only)
    """
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized with level {logging.getLevelName(log_level)}")
    if log_file:
        logger.info(f"Logs will be saved to {log_file}")



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a machine translation model using GRPO")
    parser.add_argument(
        "--model_name",
        type=str,
        default="sbintuitions/sarashina2.2-3b-instruct-v0.1",
        help="Name or path of the model to use for translation"
    )
    parser.add_argument(
        "--train_datasets",
        type=str,
        default="dataset/wmt.en-ja/newstest2021.en-ja.all.xml,dataset/wmt.en-ja/wmttest2022.en-ja.all.xml,dataset/wmt.en-ja/wmttest2023.en-ja.all.xml",
        help="Comma-separated list of training dataset paths"
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="dataset/wmt.en-ja/wmttest2024.en-ja.all.xml",
        help="Path to the test dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/pg/grpo/aw_grpo_results",
        help="Directory to save the model and results"
    )
    parser.add_argument(
        "--reward_functions",
        type=str,
        default="combined",
        help="Comma-separated list of reward functions to use"
    )
    parser.add_argument(
        "--reward_weights",
        type=str,
        default="bleu=0.05,readability=0.35,bleurt=0.6",
        help="Comma-separated list of weight assignments for the combined reward"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length for the model"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=128,
        help="LoRA rank for parameter-efficient fine-tuning"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="Number of generations for GRPO"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-6,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=150,
        help="Save checkpoint every X steps"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=200,
        help="Maximum length for generated completions"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Fraction of training data to use for validation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--bleurt_model",
        type=str,
        default="lucadiliello/BLEURT-20-D12",
        help="BLEURT model to use for evaluation"
    )

    return parser.parse_args()

def setup_model_and_tokenizer(args):
    """
    Set up the model and tokenizer for training.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {args.model_name}")
    
    # Log GPU information
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA current device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    else:
        logger.warning("CUDA is not available. Training will be slow.")
    
    try:
        # Load the model and tokenizer
        logger.info("Initializing model and tokenizer with FastLanguageModel")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
            fast_inference=True,
            max_lora_rank=args.lora_rank,
            gpu_memory_utilization=0.7,
        )
        
        # Set up LoRA for parameter-efficient fine-tuning
        logger.info(f"Setting up LoRA with rank {args.lora_rank}")
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj", "k_proj", 
            ],
            lora_alpha=args.lora_rank,
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
        
        logger.info("Model and tokenizer setup completed successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error setting up model and tokenizer: {str(e)}", exc_info=True)
        raise

def train_with_grpo(model, tokenizer, train_dataset, val_dataset, args):
    """
    Train the model using GRPO with the specified reward functions.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        train_dataset: Dataset for training
        val_dataset: Dataset for validation
        args: Command line arguments
        
    Returns:
        Trained GRPOTrainer
    """
    logger.info("Setting up GRPO training")
    
    # Get reward functions
    reward_function_names = args.reward_functions.split(',')
    reward_funcs = []
    
    for name in reward_function_names:
        try:
            reward_func = RewardRegistry.get(name.strip())
            reward_funcs.append(reward_func)
            logger.info(f"Using reward function: {name}")
        except ValueError as e:
            logger.warning(f"Warning: {e}")
    
    if not reward_funcs:
        logger.warning("No valid reward functions specified. Using combined reward.")
        reward_funcs = [RewardRegistry.get("combined")]
    
    # Parse reward weights if using combined reward
    if "combined" in reward_function_names:
        weights = parse_reward_weights(args.reward_weights)
        logger.info(f"Using reward weights: {weights}")
    
    # Log dataset information
    logger.info(f"Training dataset size: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    else:
        logger.info("No validation dataset provided")
    
    # Set up training arguments
    logger.info("Configuring GRPO training arguments")
    train_args = GRPOConfig(
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        temperature=args.temperature,
        max_prompt_length=args.max_seq_length,
        use_vllm=True,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        max_grad_norm=0.1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
    )
    
    # Create output directory if it doesn't exist
    logger.info(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments for reproducibility
    args_path = os.path.join(args.output_dir, "args.json")
    logger.info(f"Saving arguments to {args_path}")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize trainer
    logger.info("Initializing GRPO trainer")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if val_dataset else None,
        reward_funcs=reward_funcs,
        args=train_args,
    )
    
    # Train the model
    logger.info("Starting GRPO training...")
    trainer.train()
    logger.info("GRPO training completed")
    
    return trainer

def evaluate_model(model, tokenizer, test_dataset, args, timestamp: str = None):
    """
    Evaluate the trained model on the test dataset.
    
    Args:
        model: The trained language model
        tokenizer: The tokenizer
        test_dataset: Dataset for testing
        args: Command line arguments
        timestamp: Optional timestamp to use for filenames
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info("Evaluating model on test dataset...")
    
    # Get sources and references
    sources = test_dataset["source"]
    references = test_dataset["reference"]
    
    logger.info(f"Test dataset size: {len(sources)} examples")
    
    # Generate translations
    translations = []
    for i in tqdm(range(len(sources)), desc="Generating translations"):
        source = sources[i]
        prompt = create_translation_prompt(source)
        
        # Format prompt using the chat template
        logger.debug(f"Creating prompt for example {i}")
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        input_length = inputs.input_ids.shape[-1]
        
        # Generate translation
        logger.debug(f"Generating translation for example {i}")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_completion_length,
                temperature=0.1,  # Lower temperature for more deterministic outputs
            )
        
        # Extract generated text
        generated_ids = output_ids[0][input_length:]
        translation = tokenizer.decode(generated_ids, skip_special_tokens=True)
        translations.append(translation)
        
        # Log a few examples
        if i < 3 or i % 100 == 0:
            logger.info(f"Example {i}:")
            logger.info(f"  Source: {source}")
            logger.info(f"  Reference: {references[i]}")
            logger.info(f"  Translation: {translation}")
    
    # Calculate rewards
    
    logger.info("Calculating readability scores...")
    readability_scores = RewardRegistry.get("readability")(translations)
    
    logger.info(f"Calculating BLEURT scores using model: {args.bleurt_model}...")
    bleurt_scores = RewardRegistry.get("bleurt")(
        translations, 
        references, 
        bleurt_model=args.bleurt_model
    )
    # Calculate average scores
    avg_readability = sum(readability_scores) / len(readability_scores)
    avg_bleurt = sum(bleurt_scores) / len(bleurt_scores)
    # Create evaluation results
    logger.info("Creating evaluation results...")
    results = {
        "model": args.model_name,
        "test_dataset": args.test_dataset,
        "metrics": {
            "readability": {
                "average": avg_readability,
                "scores": readability_scores
            },
            "bleurt": {
                "average": avg_bleurt,
                "scores": bleurt_scores
            },
        },
        "examples": [
            {
                "source": sources[i],
                "reference": references[i],
                "translation": translations[i],
                "readability": readability_scores[i],
                "bleurt": bleurt_scores[i],
            }
            for i in range(min(10, len(sources)))  # Include first 10 examples
        ]
    }
    
    # Use provided timestamp or generate a new one
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"aw_grpo_eval_{timestamp}.json"
    
    # Local path
    local_output_path = os.path.join(args.output_dir, filename)
    
    # Save locally
    logger.info(f"Saving evaluation results to {local_output_path}")
    with open(local_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation results saved locally to {local_output_path}")
    logger.info(f"Readability: {avg_readability:.2f}")
    logger.info(f"BLEURT: {avg_bleurt:.4f}")

    
    return results


def main():
    """Main function."""
    args = parse_args()
    
    # Generate a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, f"aw_grpo_{timestamp}.log")
    setup_logging(log_level=logging.INFO, log_file=log_file)
    
    logger.info(f"Starting GRPO machine translation training with timestamp {timestamp}")
    logger.info(f"Command line arguments: {args}")
    
    # Set random seed for reproducibility
    logger.info(f"Setting random seed to {args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        # Load datasets
        logger.info(f"Loading datasets from {args.train_datasets} and {args.test_dataset}")
        train_paths = args.train_datasets.split(',')
        train_dataset, test_dataset = load_datasets(train_paths, args.test_dataset)
        
        # Set up model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(args)
        
        # Prepare datasets for GRPO
        logger.info("Preparing datasets for GRPO")
        train_dataset = prepare_dataset_for_grpo(train_dataset, tokenizer)
        
        # Split training data into train and validation sets
        logger.info(f"Splitting dataset with validation size {args.val_size}")
        train_dataset, val_dataset = split_dataset(
            train_dataset, 
            test_size=args.val_size, 
            seed=args.seed
        )
        
        # Train the model
        trainer = train_with_grpo(model, tokenizer, train_dataset, val_dataset, args)
        
        # Prepare test dataset for evaluation
        if test_dataset:
            logger.info("Preparing test dataset for evaluation")
            test_dataset = prepare_dataset_for_grpo(test_dataset, tokenizer)
            
            # Evaluate the model
            evaluation_results = evaluate_model(model, tokenizer, test_dataset, args, timestamp)
        
        logger.info("Training and evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()