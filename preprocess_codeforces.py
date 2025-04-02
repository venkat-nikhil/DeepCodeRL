import os
import json
import pandas as pd
import logging
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple, Union

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def extract_problem_description(row: pd.Series) -> str:
    """
    Extract problem description from Codeforces dataset row.
    Only uses description, input_format, output_format, and examples columns.
    
    Args:
        row: A row from the Codeforces dataset
    
    Returns:
        A formatted problem description string
    """
    problem_description = ""
    
    # Add description
    if not pd.isna(row['description']):
        problem_description += f"{row['description'].strip()}\n\n"
    
    # Add input format
    if not pd.isna(row['input_format']):
        problem_description += f"Input Format:\n{row['input_format'].strip()}\n\n"
        
    # Add output format
    if not pd.isna(row['output_format']):
        problem_description += f"Output Format:\n{row['output_format'].strip()}\n\n"
    
    # Add examples
    if not pd.isna(row['examples']):
        try:
            examples = json.loads(row['examples'])
            problem_description += "Examples:\n"
            for i, example in enumerate(examples, 1):
                problem_description += f"Example {i}:\n"
                problem_description += f"Input:\n{example.get('input', '')}\n"
                problem_description += f"Output:\n{example.get('output', '')}\n\n"
        except (json.JSONDecodeError, TypeError):
            # If examples can't be parsed, use as is
            problem_description += f"Examples:\n{row['examples']}\n\n"
    
    return problem_description.strip()

def extract_solution_code(row: pd.Series) -> Optional[str]:
    """
    Extract solution code from the accepted_solutions column.
    
    Args:
        row: A row from the Codeforces dataset
    
    Returns:
        The solution code or None if no valid solution is found
    """
    if pd.isna(row['accepted_solutions']):
        return None
    
    try:
        accepted_solutions = json.loads(row['accepted_solutions'])
        if accepted_solutions and len(accepted_solutions) > 0:
            # Use the first accepted solution's code field
            return accepted_solutions[0].get('code', '')
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse accepted_solutions for problem {row.get('id', 'unknown')}")
        return None
    
    return None

def extract_explanation(row: pd.Series) -> str:
    """
    Extract explanation from the editorial column.
    
    Args:
        row: A row from the Codeforces dataset
    
    Returns:
        The explanation string or a default explanation if none is found
    """
    if not pd.isna(row['editorial']):
        return row['editorial'].strip()
    else:
        return "This solution implements an efficient algorithm that addresses the problem requirements."

def preprocess_codeforces_data(
    csv_path: str,
    output_dir: str,
    train_ratio: float = 0.9,
    eval_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[str, str]:
    """
    Preprocess Codeforces dataset from CSV and split into train/eval sets.
    
    Args:
        csv_path: Path to the Codeforces CSV file
        output_dir: Directory to save processed data
        train_ratio: Ratio of data to use for training
        eval_ratio: Ratio of data to use for evaluation
        seed: Random seed for splitting
    
    Returns:
        Tuple of (train_path, eval_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    logger.info(f"Reading Codeforces data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} problems from Codeforces dataset")
    
    # Process each row into a standard format
    processed_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing problems"):
        # Extract problem description using only specified columns
        problem_description = extract_problem_description(row)
        
        # Extract solution code from accepted_solutions
        solution = extract_solution_code(row)
        if not solution:
            # Skip problems without a solution
            continue
        
        # Extract explanation from editorial
        explanation = extract_explanation(row)
        
        processed_data.append({
            "id": row['id'] if not pd.isna(row['id']) else f"problem_{len(processed_data)}",
            "title": row['title'] if not pd.isna(row['title']) else "",
            "problem": problem_description,
            "solution": solution,
            "explanation": explanation
        })
    
    logger.info(f"Successfully processed {len(processed_data)} problems")
    
    # Split into train and eval datasets
    train_data, eval_data = train_test_split(
        processed_data, 
        test_size=eval_ratio,
        train_size=train_ratio,
        random_state=seed
    )
    
    logger.info(f"Split into {len(train_data)} training and {len(eval_data)} evaluation examples")
    
    # Save to JSON files
    train_path = os.path.join(output_dir, "train.json")
    eval_path = os.path.join(output_dir, "eval.json")
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(eval_path, 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    logger.info(f"Saved training data to {train_path}")
    logger.info(f"Saved evaluation data to {eval_path}")
    
    return train_path, eval_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Codeforces dataset for DeepCodeRL")
    parser.add_argument("--csv_path", type=str, required=True, 
                      help="Path to Codeforces CSV file")
    parser.add_argument("--output_dir", type=str, default="data/processed/", 
                      help="Directory to save processed data")
    parser.add_argument("--train_ratio", type=float, default=0.9, 
                      help="Ratio of data to use for training")
    parser.add_argument("--eval_ratio", type=float, default=0.1, 
                      help="Ratio of data to use for evaluation")
    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed for splitting")
    
    args = parser.parse_args()
    
    preprocess_codeforces_data(
        args.csv_path,
        args.output_dir,
        args.train_ratio,
        args.eval_ratio,
        args.seed
    )