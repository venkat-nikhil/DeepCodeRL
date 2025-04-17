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
from typing import Dict, List, Optional, Tuple, Union

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class CodeDataset(Dataset):
    """
    Custom dataset for code generation tasks.
    
    This dataset handles loading and preprocessing of competitive programming problems 
    and their solutions for supervised fine-tuning.
    """
    
    def __init__(
        self, 
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        problem_prefix: str = "PROBLEM:\n",
        solution_prefix: str = "\nSOLUTION:\n",
        explanation_prefix: str = "\nEXPLANATION:\n",
        mode: str = "train",
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the JSON file containing the dataset
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
        with open(data_path, 'r', encoding='utf-8') as f:
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


# Define a custom Trainer that works with distributed models
class DistributedModelTrainer(Trainer):
    """
    A custom trainer that doesn't try to move models with device_map="auto"
    """
    def _move_model_to_device(self, model, device):
        # Check if the model is using device_map
        if hasattr(model, "hf_device_map") and model.hf_device_map:
            logger.info("Model is already distributed using device_map, skipping device placement")
            return model
        # Otherwise use the standard behavior
        return super()._move_model_to_device(model, device)


def train(args):
    """
    Train the model using supervised fine-tuning.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Tuple of (trainer, model)
    """
    set_seed(args.seed)
    
    # Initialize wandb for tracking experiments
    if hasattr(args, 'use_wandb') and args.use_wandb:
        wandb.init(project="deepcoderl", name=f"sft_{args.run_name}")
    
    # Memory optimization: clear CUDA cache before loading model and tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="right",
        use_fast=True,
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimizations
    logger.info(f"Loading base model: {args.model_name}")
    if args.use_fp16 and torch.cuda.is_available():
        torch_dtype = torch.float16
    elif args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    
    if hasattr(args, 'use_8bit') and args.use_8bit:
        logger.info("Using 8-bit quantization")
        model_kwargs["load_in_8bit"] = True
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs,
    )
    
    # Enable gradient checkpointing to save memory
    if hasattr(args, 'gradient_checkpointing') and args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    
    # Apply LoRA if specified (for more efficient fine-tuning)
    if hasattr(args, 'use_lora') and args.use_lora:
        logger.info("Applying LoRA for parameter-efficient fine-tuning")
        
        # Prepare model for k-bit training if using quantization
        if hasattr(args, 'use_8bit') and args.use_8bit:
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
    train_dataset = CodeDataset(
        data_path=args.train_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mode="train"
    )
    
    eval_dataset = CodeDataset(
        data_path=args.eval_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mode="eval"
    )
    
    # Configure training arguments
    # Ensure save_steps is a multiple of eval_steps when using load_best_model_at_end
    save_steps = args.save_steps
    eval_steps = args.eval_steps
    
    if save_steps % eval_steps != 0:
        # Adjust save_steps to be the nearest multiple of eval_steps
        save_steps = eval_steps * round(save_steps / eval_steps)
        logger.info(f"Adjusted save_steps to {save_steps} to be a multiple of eval_steps ({eval_steps})")
    
    training_args = TrainingArguments(
        output_dir=args.sft_output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler,
        learning_rate=args.learning_rate,
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.use_fp16,
        bf16=args.bf16,
        report_to="wandb" if (hasattr(args, 'use_wandb') and args.use_wandb) else "none",
        optim=args.optimizer,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Memory optimization options
        dataloader_num_workers=args.num_workers if hasattr(args, 'num_workers') else 0,
        dataloader_pin_memory=True,
        torch_compile=(hasattr(args, 'torch_compile') and args.torch_compile),  # PyTorch 2.0+ optimization
    )
    
    # Initialize trainer - using our special trainer class
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're using causal language modeling (not masked)
    )
    
    # Always use our distributed model trainer to handle device mapping correctly
    trainer = DistributedModelTrainer(
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
    final_model_path = os.path.join(args.sft_output_dir, "final_model")
    logger.info(f"Saving model to {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Log final metrics
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    if hasattr(args, 'use_wandb') and args.use_wandb:
        wandb.log({"final_eval": eval_results})
    
    logger.info("Training completed!")
    
    return trainer, model