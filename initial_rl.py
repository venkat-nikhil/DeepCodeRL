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

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized examples in batches.
    """
    # Extract items that need to be stacked
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Keep non-tensor items as lists without trying to collate them
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'problem_id': [item['problem_id'] for item in batch],
        'problem_text': [item['problem_text'] for item in batch],
        'full_prompt': [item['full_prompt'] for item in batch],
        'reference_solution': [item['reference_solution'] for item in batch],
        'examples': [item['examples'] for item in batch]
    }
    
    return result

def calculate_reward(generated_code: str, examples) -> float:
    """
    Calculate reward for the generated code.
    
    Args:
        generated_code: The generated Python code
        examples: Examples for testing
        
    Returns:
        float: Reward value
    """
    try:
        # Process examples safely to handle different formats from batching
        valid_examples = []
        if isinstance(examples, list):
            # First, check if the list itself contains valid examples
            if examples and isinstance(examples[0], dict) and ('input' in examples[0] or 'output' in examples[0]):
                valid_examples = examples
            else:
                # Examples might be a nested structure from batching
                for ex in examples:
                    if isinstance(ex, dict) and ('input' in ex or 'output' in ex):
                        valid_examples.append(ex)
                    elif isinstance(ex, list) and ex:
                        # If we have a list of examples (from batching), take the first batch
                        if isinstance(ex[0], dict):
                            valid_examples.extend([e for e in ex if isinstance(e, dict) and ('input' in e or 'output' in e)])
                        break
        elif isinstance(examples, dict):
            valid_examples = [examples]
            
        if not valid_examples:
            logger.warning("No valid examples found for reward calculation")
            return 0.0
            
        test_inputs = []
        test_input = scrape_and_run_code(text=generated_code, examples=valid_examples)
        test_inputs.append(test_input)
        
        tester = MultiProcessorEvaluator(
            command_prefix=['python','-c'],
            max_workers=1,
            timeout=2.0
        )
        results = tester.run(test_inputs)
        
        # Process results directly instead of using get_batch_run_scores
        if not results:
            return 0.0
            
        # Handle different possible result formats
        try:
            if isinstance(results[0], list):
                # If results is a list of lists
                flat_results = results[0]
                if all(isinstance(item, bool) for item in flat_results):
                    score = sum(flat_results) / len(flat_results) if flat_results else 0.0
                else:
                    # Try to extract booleans from tuples or other structures
                    bools = []
                    for item in flat_results:
                        if isinstance(item, bool):
                            bools.append(item)
                        elif isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], bool):
                            bools.append(item[0])
                    score = sum(bools) / len(bools) if bools else 0.0
            elif isinstance(results[0], bool):
                # If results is a list of booleans
                score = sum(results) / len(results)
            else:
                score = 0.0
        except Exception as e:
            logger.warning(f"Error processing evaluation results: {e}")
            score = 0.0
            
        return max(0.0, float(score))
    except Exception as e:
        logger.warning(f"Error in calculate_reward: {e}")
        return 0.0


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
        padding_side="left",  # Set to left padding for decoder-only models
        use_fast=True,
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimizations
    logger.info(f"Loading model from: {args.sft_model_path if args.sft_model_path else args.model_name}")
    
    # Configure model loading parameters with added FP16 stability
    if hasattr(args, 'use_fp16') and args.use_fp16:
        logger.info("Using FP16 mixed precision with added stability measures")
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
        
    model_kwargs = {
        "torch_dtype": torch_dtype,
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
    
    # Ensure all parameters are trainable
    for param in model.parameters():
        param.requires_grad = True
    
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
    batch_size = args.batch_size  # Use smaller batch size for RL to conserve memory
    num_workers = args.num_workers if hasattr(args, 'num_workers') else 0
    
    logger.info(f"Using batch size of {batch_size} for RL training")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    
    # Set up optimizer
    learning_rate = args.rl_learning_rate if hasattr(args, 'rl_learning_rate') else 1e-6
    weight_decay = args.weight_decay if hasattr(args, 'weight_decay') else 0.01
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
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
                # Generate code completions with safer parameters for FP16
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,  # Add top_k to prevent extreme probabilities
                    temperature=0.8,  # Slightly higher temperature for more stable sampling
                    return_dict_in_generate=True,
                    output_scores=True,
                    min_length=5,  # Set minimum length to avoid empty generations
                    bad_words_ids=None,
                    num_return_sequences=1,  # Ensure we get exactly one sequence per input
                    # repetition_penalty=1.1
                )
                
                generated_sequences = outputs.sequences
                
                # Calculate rewards for the generated code
                rewards = []
                
                # Open log file for current epoch to save generated code
                log_filename = f"logs/model-output-epoch-{epoch+1}.txt"
                log_file = open(log_filename, "a", encoding="utf-8")
                
                for i, sequence in enumerate(generated_sequences):
                    # Find where the input ends
                    batch_idx = i % input_ids.size(0)
                    input_length = input_ids[batch_idx].size(0)
                    
                    # Get only the newly generated part
                    generated_tokens = sequence[input_length:]
                    generated_code = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    # Get the specific examples for this sequence
                    sequence_examples = batch_examples[batch_idx] if isinstance(batch_examples, list) else batch_examples
                    
                    # Calculate reward
                    reward = calculate_reward(generated_code, examples=sequence_examples)
                    rewards.append(reward)
                    
                    # Get problem ID and truncated prompt for logging
                    batch_idx = i % input_ids.size(0)
                    problem_id = batch["problem_id"][batch_idx] if "problem_id" in batch else f"problem_{total_steps}_{i}"
                    problem_text = batch["problem_text"][batch_idx] if "problem_text" in batch else "Unknown problem"
                    prompt_text = batch["full_prompt"][batch_idx] if "full_prompt" in batch else "Unknown prompt"
                    
                    # Truncate problem text for readability in logs
                    problem_text_short = problem_text[:200] + "..." if len(problem_text) > 200 else problem_text
                    
                    # Write to log file immediately with flush to ensure real-time logging
                    log_file.write(f"===== SAMPLE {total_steps}_{i} =====\n")
                    log_file.write(f"PROBLEM ID: {problem_id}\n")
                    log_file.write(f"PROBLEM TEXT: {problem_text_short}\n")
                    log_file.write(f"FULL PROMPT: {prompt_text}\n")
                    log_file.write(f"GENERATED CODE:\n{generated_code}\n")
                    log_file.write(f"REWARD: {reward}\n")
                    log_file.write("="*50 + "\n\n")
                    log_file.flush()  # Ensure immediate write to disk
                
                # Close log file
                log_file.close()
                
                # Convert rewards to tensor
                rewards = torch.tensor(rewards, device=model.device)
            
            # STEP 2: Compute REINFORCE loss
            loss = torch.tensor(0.0, device=model.device, requires_grad=True)
            valid_sequences = 0
            
            for i, sequence in enumerate(generated_sequences):
                # Skip sequences with zero reward or too small reward (no learning signal)
                # Higher threshold for FP16 to avoid numerical instability
                min_reward_threshold = 0.01 if torch_dtype == torch.float16 else 0.0
                if rewards[i] <= min_reward_threshold:
                    continue
                
                valid_sequences += 1
                
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
                
                # FP16-safe log probability computation
                # Add a small epsilon to prevent numerical instability
                epsilon = 1e-8 if torch_dtype == torch.float16 else 1e-10
                
                # Apply log softmax with stability clipping
                logits_for_generated_stable = logits_for_generated
                
                # Add epsilon before log_softmax to prevent -inf values
                log_probs = torch.nn.functional.log_softmax(logits_for_generated_stable, dim=-1)
                
                # Replace any potential NaN or Inf values for safety
                log_probs = torch.nan_to_num(log_probs, nan=-100.0, posinf=-100.0, neginf=-100.0)
                
                # Get the log probability of the tokens that were actually generated
                # Ensure index is valid
                max_length = min(generated_part.size(1), logits_for_generated.size(1))
                valid_generated_part = generated_part[:, :max_length].unsqueeze(-1)
                
                log_prob_generated = torch.gather(
                    log_probs[:, :max_length],
                    dim=-1,
                    index=valid_generated_part
                ).squeeze(-1)
                
                # Clip extreme negative values to prevent numerical instability
                log_prob_generated = torch.clamp(log_prob_generated, min=-100.0, max=0.0)
                
                # Sum log probs to get sequence log probability
                seq_log_prob = log_prob_generated.sum()
                
                # Convert reward to tensor with proper device
                reward_tensor = rewards[i].clone().detach()
                
                # Compute policy gradient loss: negative log probability times reward
                # (Negative because we're doing gradient descent but want to maximize reward)
                sequence_loss = -seq_log_prob * reward_tensor
                loss = loss + sequence_loss
            
            # Normalize loss by the number of valid sequences
            if valid_sequences > 0:
                loss = loss / valid_sequences
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Backpropagation - make sure loss requires grad
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Optimization step
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                epoch_reward += rewards.mean().item()
                num_batches += 1
            else:
                logger.warning("No valid sequences in batch - skipping update")
            
            # Update progress bar
            total_steps += 1
            avg_reward = epoch_reward / max(num_batches, 1)
            progress_bar.set_postfix({
                "loss": epoch_loss / max(num_batches, 1),
                "reward": avg_reward,
                "valid_seqs": valid_sequences
            })
            
            # Periodically save checkpoint
            if total_steps % save_steps == 0:
                checkpoint_path = os.path.join(args.rl_output_dir, f"checkpoint-{total_steps}")
                os.makedirs(checkpoint_path, exist_ok=True)
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # End of epoch logging
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        avg_epoch_reward = epoch_reward / max(num_batches, 1)
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