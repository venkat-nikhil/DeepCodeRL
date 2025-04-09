import os
import json
import logging
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from tqdm import tqdm
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import wandb
import argparse
from typing import Dict, List, Optional, Tuple, Union

# Import the preprocessing module
from preprocess_codeforces import preprocess_codeforces_data

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class CodeForcesDataset(Dataset):
    """
    Custom dataset for Codeforces competitive programming problems.
    
    Handles loading and preprocessing of Codeforces problems 
    and their solutions for supervised fine-tuning.
    """
    
    def __init__(
        self, 
        data_path: str,
        tokenizer,
        max_length: int = 1024,  # Reduced from 2048 to save memory
        problem_prefix: str = "PROBLEM:\n",
        solution_prefix: str = "\nSOLUTION:\n",
        explanation_prefix: str = "\nEXPLANATION:\n",
        mode: str = "train",
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the preprocessed JSON file containing the dataset
            tokenizer: Tokenizer to use for encoding inputs
            max_length: Maximum sequence length
            problem_prefix: Prefix for problem statements
            solution_prefix: Prefix for solutions
            explanation_prefix: Prefix for explanations
            mode: 'train' or 'eval' mode
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.problem_prefix = problem_prefix
        self.solution_prefix = solution_prefix
        self.explanation_prefix = explanation_prefix
        self.mode = mode
        
        # Load data from JSON file
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} examples from {data_path} for {mode}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a tokenized example for training.
        
        The format is: PROBLEM: <problem_description> SOLUTION: <solution> EXPLANATION: <explanation>
        """
        item = self.data[idx]
        
        # Construct the full text with problem, solution, and explanation
        full_text = (
            self.problem_prefix + item["problem"] +
            self.explanation_prefix + item["explanation"] +
            self.solution_prefix + item["solution"]
        )
        
        # Tokenize the full text
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create attention mask and labels
        input_ids = encodings.input_ids[0]
        attention_mask = encodings.attention_mask[0]
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        
        if self.mode == "train":
            # Mask out the loss for the problem part (we only want to predict solution and explanation)
            problem_tokens = self.tokenizer(
                self.problem_prefix + item["problem"] + self.explanation_prefix + item["explanation"],
                return_tensors="pt"
            ).input_ids[0]
            problem_length = len(problem_tokens)
            labels[:problem_length] = -100  # -100 is the ignore index for CrossEntropyLoss
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def train(args):
    """
    Train the model using supervised fine-tuning.
    
    Args:
        args: Command-line arguments
    """
    set_seed(args.seed)
    
    # Initialize wandb for tracking experiments
    if args.use_wandb:
        wandb.init(project="deepcoderl", name=f"sft_{args.run_name}")
    
    # Create processed data directory if it doesn't exist
    os.makedirs(args.processed_data_dir, exist_ok=True)
    
    # Process data if not already done
    train_path = os.path.join(args.processed_data_dir, "train.json")
    eval_path = os.path.join(args.processed_data_dir, "eval.json")
    
    if not (os.path.exists(train_path) and os.path.exists(eval_path)):
        logger.info(f"Processed data not found. Preprocessing from {args.csv_path}")
        train_path, eval_path = preprocess_codeforces_data(
            args.csv_path,
            args.processed_data_dir,
            train_ratio=args.train_ratio,
            eval_ratio=args.eval_ratio,
            seed=args.seed
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="right",
        use_fast=True,
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Memory optimization: clear CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load model with 8-bit quantization if specified
    logger.info(f"Loading base model: {args.model_name}")
    model_kwargs = {
        "torch_dtype": torch.float16 if args.use_fp16 else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    
    if args.use_8bit:
        logger.info("Using 8-bit quantization")
        model_kwargs["load_in_8bit"] = True
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs,
    )
    
    # Enable gradient checkpointing to save memory
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    
    # Apply LoRA if specified (for more efficient fine-tuning)
    if args.use_lora:
        logger.info("Applying LoRA for parameter-efficient fine-tuning")
        
        # Prepare model for k-bit training if using quantization
        if args.use_8bit:
            logger.info("Preparing model for 8-bit training")
            model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Prepare datasets
    train_dataset = CodeForcesDataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mode="train"
    )
    
    eval_dataset = CodeForcesDataset(
        data_path=eval_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mode="eval"
    )
    
    # Configure training arguments
    # Ensure save_steps is a multiple of eval_steps when using load_best_model_at_end
    if args.save_steps % args.eval_steps != 0:
        # Adjust save_steps to be the nearest multiple of eval_steps
        args.save_steps = args.eval_steps * round(args.save_steps / args.eval_steps)
        logger.info(f"Adjusted save_steps to {args.save_steps} to be a multiple of eval_steps ({args.eval_steps})")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.use_fp16,
        report_to="wandb" if args.use_wandb else "none",
        optim=args.optimizer,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Memory optimization options
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        torch_compile=args.torch_compile,  # PyTorch 2.0+ optimization
    )
    
    # Initialize trainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're using causal language modeling (not masked)
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {os.path.join(args.output_dir, 'final_model')}")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    # Log final metrics
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    if args.use_wandb:
        wandb.log({"final_eval": eval_results})
    
    logger.info("Training completed!")
    
    return trainer, model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DeepCodeRL: Supervised Fine-Tuning Phase for Codeforces Dataset")
    
    # Data arguments
    parser.add_argument("--csv_path", type=str, required=True, 
                      help="Path to Codeforces CSV file")
    parser.add_argument("--processed_data_dir", type=str, default="data/processed/", 
                      help="Directory to save processed data")
    parser.add_argument("--train_ratio", type=float, default=0.9, 
                      help="Ratio of data to use for training")
    parser.add_argument("--eval_ratio", type=float, default=0.1, 
                      help="Ratio of data to use for evaluation")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-coder-1.5b", 
                      help="Pretrained model name or path")
    parser.add_argument("--max_length", type=int, default=1024,  # Reduced from 2048 to save memory
                      help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output/sft", 
                      help="Output directory for model checkpoints")
    parser.add_argument("--batch_size", type=int, default=1,  # Reduced from 4 to save memory
                      help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,  # Increased from 4 to save memory
                      help="Number of steps for gradient accumulation")
    parser.add_argument("--num_epochs", type=float, default=3, 
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                      help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                      help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, 
                      help="Number of warmup steps")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", 
                      help="Learning rate scheduler type")
    parser.add_argument("--logging_steps", type=int, default=5, 
                      help="Logging steps")
    parser.add_argument("--eval_steps", type=int, default=5, 
                      help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=5,  # Set to match eval_steps
                      help="Steps between checkpoint saves")
    parser.add_argument("--save_total_limit", type=int, default=300, 
                      help="Maximum number of checkpoints to save")
    parser.add_argument("--optimizer", type=str, default="adamw_torch", 
                      help="Optimizer to use")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", 
                      help="Whether to use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, 
                      help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                      help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                      help="LoRA dropout probability")
    
    # Memory optimization arguments
    parser.add_argument("--use_8bit", action="store_true",
                      help="Whether to use 8-bit quantization")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                      help="Whether to use gradient checkpointing")
    parser.add_argument("--torch_compile", action="store_true",
                      help="Whether to use torch.compile() for PyTorch 2.0+ optimization")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed")
    parser.add_argument("--use_fp16", action="store_true", 
                      help="Whether to use mixed precision training")
    parser.add_argument("--use_wandb", action="store_true", 
                      help="Whether to use Weights & Biases for logging")
    parser.add_argument("--run_name", type=str, default="deepcoderl_sft_codeforces", 
                      help="Run name for tracking")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the training
    trainer, model = train(args)