import os
import json
import logging
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig
)
import ast
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
from eval_framework.eval import *
from parser import scrape_and_run_code

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def check_syntax_validity(code: str) -> float:
    """
    Check if the generated code has valid Python syntax.
    
    Args:
        code: The generated Python code
        
    Returns:
        float: 1.0 if syntax is valid, 0.0 otherwise
    """
    try:
        ast.parse(code)  # This will throw an error if the code is invalid
        return 1.0  # Reward for valid code
    except SyntaxError:
        return 0.0  # Penalty for invalid code
    except Exception as e:
        logger.warning(f"Error parsing code: {e}")
        return 0.0  # Penalty for other errors



class RLCodeDataset(Dataset):
    """
    Dataset for RL training of code generation models.
    This dataset is designed for the Initial RL phase with reduced context.
    """
    
    def __init__(
        self, 
        data_path: str,
        tokenizer,
        max_length: int = 512,  # Reduced context window as per proposal
        problem_prefix: str = "PROBLEM:\n",
        solution_prefix: str = "\nSOLUTION:\n",
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the JSON file containing the dataset
            tokenizer: Tokenizer to use for encoding inputs
            max_length: Maximum sequence length (reduced context for RL)
            problem_prefix: Prefix for problem statements
            solution_prefix: Prefix for solutions
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.problem_prefix = problem_prefix
        self.solution_prefix = solution_prefix
        
        # Load data from JSON file
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} examples from {data_path} for RL training")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get an example for RL training.
        
        Returns a dictionary containing the tokenized input prompt and other metadata.
        """
        item = self.data[idx]
        
        # Get problem description with prefix
        problem_text = self.problem_prefix + item["problem"]
        
        # Get the full prompt for generation
        full_prompt = problem_text + self.solution_prefix
        
        # Tokenize the prompt
        encodings = self.tokenizer(
            full_prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings.input_ids[0],
            "attention_mask": encodings.attention_mask[0],
            "problem_id": item["id"],
            "problem_text": problem_text,
            "full_prompt": full_prompt,
            "reference_solution": item["solution"],  # Reference solution for debugging
            "examples": item["examples"]
        }

def calculate_reward(generated_code: str, examples: List[Dict[str, str]]) -> float:
    """
    Calculate reward for the generated code.
    Currently just checks syntax validity, but can be extended.
    
    Args:
        generated_code: The generated Python code
        
    Returns:
        float: Reward value
    """
    
    # return check_syntax_validity(generated_code)

    test_inputs = []
    test_inputs.append(scrape_and_run_code(text=generated_code, examples=examples))

    tester = MultiProcessorEvaluator(
        command_prefix=['python','-c'],  # or None to auto‑use sys.executable
        max_workers=1,
        timeout=2.0
    )
    results = tester.run(test_inputs)
    reward = float(results[0]) if results and results[0] is not None else 0.0
    return max(0.0, reward)

def train_rl(args):
    """
    Train the model using a simple REINFORCE algorithm (policy gradient approach).
    
    This implements the Initial RL phase with reduced context as described in the proposal.
    
    Args:
        args: Command-line arguments
        
    Returns:
        The trained model
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create processed data directory if it doesn't exist
    os.makedirs(args.processed_data_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.sft_model_path if args.sft_model_path else args.model_name,
        padding_side="right",
        use_fast=True,
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimizations
    logger.info(f"Loading model from: {args.sft_model_path if args.sft_model_path else args.model_name}")
    
    # Configure model loading parameters
    model_kwargs = {
        "torch_dtype": torch.float16 if hasattr(args, 'use_fp16') and args.use_fp16 else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    
    # Add quantization configuration
    if hasattr(args, 'use_8bit') and args.use_8bit:
        logger.info("Using 8-bit quantization with CPU offloading enabled")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offloading for modules that don't fit in GPU
        )
    
    # Load model from SFT phase or pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path if args.sft_model_path else args.model_name,
        **model_kwargs,
    )
    
    # Enable gradient checkpointing for memory efficiency if specified
    if hasattr(args, 'gradient_checkpointing') and args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    
    # Prepare dataset with reduced context window
    max_length = args.rl_max_length if hasattr(args, 'rl_max_length') else 512
    logger.info(f"Using context window of {max_length} tokens for RL phase")
    
    train_dataset = RLCodeDataset(
        data_path=args.train_data_path,
        tokenizer=tokenizer,
        max_length=max_length,  # Reduced context for RL phase
    )
    
    # Create dataloader
    batch_size = min(args.batch_size, 2)  # Use smaller batch size for RL to conserve memory
    num_workers = args.num_workers if hasattr(args, 'num_workers') else 0
    
    logger.info(f"Using batch size of {batch_size} for RL training")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    # Set up optimizer
    learning_rate = args.rl_learning_rate if hasattr(args, 'rl_learning_rate') else 1e-6
    weight_decay = args.weight_decay if hasattr(args, 'weight_decay') else 0.01
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Training loop
    logger.info("Starting RL training phase with reduced context...")
    model.train()
    
    total_steps = 0
    best_reward = 0.0
    max_grad_norm = args.max_grad_norm if hasattr(args, 'max_grad_norm') else 1.0
    save_steps = getattr(args, 'save_steps', 200)
    num_epochs = int(args.num_epochs) if hasattr(args, 'num_epochs') else 3
    max_new_tokens = args.max_new_tokens if hasattr(args, 'max_new_tokens') else 512
    
    logger.info(f"Training for {num_epochs} epochs with learning rate {learning_rate}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_reward = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move tensors to the right device
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            batch_examples = batch["examples"]
            
            # STEP 1: Generate code using the current model
            with torch.no_grad():
                # Generate code completions
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.7,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                
                generated_sequences = outputs.sequences
                
                # Calculate rewards for the generated code
                rewards = []
                for i, sequence in enumerate(generated_sequences):
                    # Find where the input ends
                    input_length = input_ids[i % input_ids.size(0)].size(0)
                    
                    # Get only the newly generated part
                    generated_tokens = sequence[input_length:]
                    generated_code = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    # Calculate reward (syntax validity)
                    reward = calculate_reward(generated_code, examples=batch_examples)
                    rewards.append(reward)
                
                # Convert rewards to tensor
                rewards = torch.tensor(rewards, device=model.device)
            
            # STEP 2: Compute REINFORCE loss
            loss = 0.0
            for i, sequence in enumerate(generated_sequences):
                # Skip sequences with zero reward (no learning signal)
                if rewards[i] == 0.0:
                    continue
                
                # Get input length
                input_length = input_ids[i % input_ids.size(0)].size(0)
                
                # Get only the generated tokens
                generated_part = sequence[input_length:].unsqueeze(0)
                
                # Prepare input for forward pass: all tokens except the last one
                model_input = sequence[:-1].unsqueeze(0)
                
                # Forward pass to get logits
                outputs = model(
                    input_ids=model_input,
                    return_dict=True
                )
                
                logits = outputs.logits
                
                # Get logits for the generated part (shifted to align)
                logits_for_generated = logits[:, input_length-1:-1]
                
                # Compute log probabilities
                log_probs = torch.nn.functional.log_softmax(logits_for_generated, dim=-1)
                
                # Get the log probability of the tokens that were actually generated
                log_prob_generated = torch.gather(
                    log_probs,
                    dim=-1,
                    index=generated_part[:, :logits_for_generated.size(1)].unsqueeze(-1)
                ).squeeze(-1)
                
                # Sum log probs to get sequence log probability
                seq_log_prob = log_prob_generated.sum()
                
                # Compute policy gradient loss: negative log probability times reward
                # (Negative because we're doing gradient descent but want to maximize reward)
                batch_loss = -seq_log_prob * rewards[i]
                loss += batch_loss
            
            # Normalize loss by batch size
            if loss > 0:  # Only if we have at least one valid sequence
                loss = loss / input_ids.size(0)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Optimization step
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                epoch_reward += rewards.mean().item()
                num_batches += 1
            
            # Update progress bar
            total_steps += 1
            avg_reward = epoch_reward / (num_batches + 1e-8)
            progress_bar.set_postfix({
                "loss": epoch_loss / (num_batches + 1e-8),
                "reward": avg_reward
            })
            
            # Periodically save checkpoint
            if total_steps % save_steps == 0:
                checkpoint_path = os.path.join(args.rl_output_dir, f"checkpoint-{total_steps}")
                os.makedirs(checkpoint_path, exist_ok=True)
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # End of epoch logging
        avg_epoch_loss = epoch_loss / (num_batches + 1e-8)
        avg_epoch_reward = epoch_reward / (num_batches + 1e-8)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_epoch_loss:.4f}, Reward: {avg_epoch_reward:.4f}")
        
        # Save model if this is the best epoch so far
        if avg_epoch_reward > best_reward:
            best_reward = avg_epoch_reward
            best_model_path = os.path.join(args.rl_output_dir, "best_model")
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            logger.info(f"Saved best model with reward {best_reward:.4f} to {best_model_path}")
    
    # Save final model
    final_model_path = os.path.join(args.rl_output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    return model