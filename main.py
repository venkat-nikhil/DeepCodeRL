import os
import argparse
import logging
import torch
from typing import Dict, Any, Optional, Tuple

from utils import setup_signal_handlers

import signal
import sys
import os
import logging

# Setup signal handlers at the beginning
setup_signal_handlers()

# Import necessary functions from individual modules
from preprocess_codeforces import preprocess_json_dataset
from SFT import train as sft_train
from initial_rl import train_rl as rl_train


# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def preprocess_data(args: argparse.Namespace) -> Tuple[str, str]:
    """
    Preprocess the data for both SFT and RL phases.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Tuple of (train_path, eval_path)
    """
    # Create processed data directory if it doesn't exist
    os.makedirs(args.processed_data_dir, exist_ok=True)
    
    # Process data if not already done
    train_path = os.path.join(args.processed_data_dir, "train.json")
    eval_path = os.path.join(args.processed_data_dir, "eval.json")
    
    if not (os.path.exists(train_path) and os.path.exists(eval_path)):
        logger.info(f"Processed data not found. Preprocessing from {args.json_path}")
        train_path, eval_path = preprocess_json_dataset(
            args.json_path,
            args.processed_data_dir,
            train_ratio=args.train_ratio,
            eval_ratio=args.eval_ratio,
            seed=args.seed
        )
    else:
        logger.info(f"Using existing processed data from {args.processed_data_dir}")
    
    return train_path, eval_path

def run_sft_phase(args: argparse.Namespace) -> str:
    """
    Run the Supervised Fine-Tuning phase.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Path to the saved SFT model
    """
    logger.info("Starting Supervised Fine-Tuning Phase...")
    
    # First ensure data is preprocessed
    train_path, eval_path = preprocess_data(args)
    
    # Add paths to args
    args.train_path = train_path
    args.eval_path = eval_path
    
    # Run SFT training
    trainer, model = sft_train(args)
    
    # Return path to saved model
    model_path = os.path.join(args.sft_output_dir, "final_model")
    logger.info(f"Supervised Fine-Tuning Phase completed. Model saved to {model_path}")
    
    return model_path

def run_initial_rl_phase(args: argparse.Namespace, sft_model_path: str) -> str:
    """
    Run the Initial RL phase with reduced context.
    
    Args:
        args: Command-line arguments
        sft_model_path: Path to the supervised fine-tuned model
        
    Returns:
        Path to the saved RL model
    """
    logger.info("Starting Initial RL Phase with reduced context...")
    
    # First ensure data is preprocessed
    train_path, _ = preprocess_data(args)
    
    # Add paths to args
    args.train_data_path = train_path
    args.sft_model_path = sft_model_path
    
    # Run RL training
    model = rl_train(args)
    
    # Return path to saved model
    model_path = os.path.join(args.rl_output_dir, "final_model")
    logger.info(f"Initial RL Phase completed. Model saved to {model_path}")
    
    return model_path

def parse_args():
    """
    Parse command-line arguments for all phases.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="DeepCodeRL: Complete Training Pipeline")
    
    # Phase control arguments
    parser.add_argument("--run_sft", action="store_true", help="Run the SFT phase")
    parser.add_argument("--run_initial_rl", action="store_true", help="Run the Initial RL phase")
    parser.add_argument("--run_all", action="store_true", help="Run all phases sequentially")
    
    # Model input/output paths
    parser.add_argument("--sft_model_path", type=str, default=None, 
                      help="Path to existing SFT model (if not running SFT phase)")
    
    # Data preprocessing arguments
    parser.add_argument("--json_path", type=str, default="data/processed_codeforces/filtered_solutions_py_decontaminated_final.json",
                      help="Path to JSON dataset file")
    parser.add_argument("--processed_data_dir", type=str, default="data/processed_sft/", 
                      help="Directory to save processed data")
    parser.add_argument("--train_ratio", type=float, default=0.9, 
                      help="Ratio of data to use for training")
    parser.add_argument("--eval_ratio", type=float, default=0.1, 
                      help="Ratio of data to use for evaluation")
    
    # SFT specific arguments
    parser.add_argument("--sft_output_dir", type=str, default="./output/sft", 
                      help="Output directory for SFT model checkpoints")
    parser.add_argument("--model_name", type=str, default="agentica-org/DeepScaleR-1.5B-Preview", 
                      help="Pretrained model name or path")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                      help="Number of steps for gradient accumulation")
    parser.add_argument("--num_epochs", type=float, default=1, 
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                      help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                      help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, 
                      help="Number of warmup steps")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", 
                      help="Learning rate scheduler type")
    parser.add_argument("--logging_steps", type=int, default=50, 
                      help="Logging steps")
    parser.add_argument("--eval_steps", type=int, default=200, 
                      help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=200,
                      help="Steps between checkpoint saves")
    parser.add_argument("--save_total_limit", type=int, default=3, 
                      help="Maximum number of checkpoints to save")
    parser.add_argument("--optimizer", type=str, default="adamw_torch", 
                      help="Optimizer to use")
    parser.add_argument("--max_length", type=int, default=1024,
                      help="Maximum sequence length")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", 
                      help="Whether to use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, 
                      help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                      help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                      help="LoRA dropout probability")
    
    # Initial RL specific arguments
    parser.add_argument("--rl_output_dir", type=str, default="./output/initial_rl", 
                      help="Output directory for RL model checkpoints")
    parser.add_argument("--rl_learning_rate", type=float, default=1e-6, 
                      help="Learning rate for RL (lower than SFT)")
    parser.add_argument("--rl_max_length", type=int, default=512,
                      help="Maximum sequence length for RL (reduced context)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                      help="Maximum number of tokens to generate in RL")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                      help="Maximum gradient norm for clipping in RL")
    
    # Memory optimization arguments
    parser.add_argument("--gradient_checkpointing", action="store_true",
                      help="Whether to use gradient checkpointing")
    parser.add_argument("--use_8bit", action="store_true",
                      help="Whether to use 8-bit quantization")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of dataloader workers")
    parser.add_argument("--torch_compile", action="store_true",
                      help="Whether to use torch.compile() for PyTorch 2.0+ optimization")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed")
    parser.add_argument("--use_fp16", action="store_true", 
                      help="Whether to use mixed precision training")
    parser.add_argument("--bf16", action="store_true",            # ← NEW
                    help="Use bfloat16 mixed precision")
    parser.add_argument("--use_wandb", action="store_true", 
                      help="Whether to use Weights & Biases for logging")
    parser.add_argument("--run_name", type=str, default="deepcoderl", 
                      help="Run name for tracking")
    
    return parser.parse_args()

def main():
    """Run the complete DeepCodeRL pipeline."""
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(args.sft_output_dir, exist_ok=True)
    os.makedirs(args.rl_output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Determine which phases to run
    run_sft = args.run_sft or args.run_all
    run_initial_rl = args.run_initial_rl or args.run_all
    
    # Track model paths
    sft_model_path = args.sft_model_path
    
    # Run SFT phase if requested
    if run_sft:
        sft_model_path = run_sft_phase(args)
    
    # Run Initial RL phase if requested
    if run_initial_rl:
        # Verify we have an SFT model path
        if sft_model_path is None:
            logger.error("No SFT model path provided. Cannot run Initial RL phase.")
            return
        
        run_initial_rl_phase(args, sft_model_path)
    
    logger.info("DeepCodeRL pipeline completed successfully.")

if __name__ == "__main__":
    main()