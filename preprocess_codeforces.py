import os
import json
import logging
import numpy as np
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def load_json_dataset(json_path: str) -> List[Dict]:
    """
    Load dataset from JSON file.
    
    Args:
        json_path: Path to the JSON file
    
    Returns:
        List of data records
    """
    logger.info(f"Loading JSON dataset from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} records from JSON dataset")
    return data

def preprocess_json_dataset(
    json_path: str,
    output_dir: str,
    train_ratio: float = 0.9,
    eval_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[str, str]:
    """
    Preprocess dataset from JSON and split into train/eval sets.
    Handles the specific fields: description, input_format, output_format, note,
    examples, editorial, and accepted_solutions.
    
    Args:
        json_path: Path to the JSON file
        output_dir: Directory to save processed data
        train_ratio: Ratio of data to use for training
        eval_ratio: Ratio of data to use for evaluation
        seed: Random seed for splitting
    
    Returns:
        Tuple of (train_path, eval_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    
    # Load the dataset
    data = load_json_dataset(json_path)
    
    # Group records by problem ID
    problem_groups = defaultdict(list)
    for record in data:
        # Use the id field for grouping
        problem_id = record.get('id', None)
        
        if problem_id is None:
            logger.warning(f"Skipping record without ID")
            continue
        
        problem_groups[problem_id].append(record)
    
    logger.info(f"Found {len(problem_groups)} unique problems with {len(data)} total solutions")
    
    # Create processed records with one solution per problem
    processed_data = []
    for problem_id, records in tqdm(problem_groups.items(), desc="Processing problems"):
        # Use the first record for problem details
        record = records[0]
        
        # Extract all required fields
        description = record.get('description', '')
        input_format = record.get('input_format', '')
        output_format = record.get('output_format', '')
        note = record.get('note', '')
        editorial = record.get('editorial', '')
        title = record.get('title', '')
        
        # Parse examples
        examples_text = ""
        if 'examples' in record and record['examples']:
            try:
                examples = json.loads(record['examples']) if isinstance(record['examples'], str) else record['examples']
                if isinstance(examples, list):
                    examples_text = "Examples:\n"
                    for i, example in enumerate(examples, 1):
                        examples_text += f"Example {i}:\n"
                        if isinstance(example, dict):
                            examples_text += f"Input:\n{example.get('input', '')}\n"
                            examples_text += f"Output:\n{example.get('output', '')}\n\n"
                        else:
                            examples_text += f"{example}\n\n"
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse examples for problem {problem_id}: {e}")
                # If we can't parse as JSON, use as raw text
                examples_text = f"Examples:\n{record['examples']}\n\n"
        
        # Combine all parts into full problem description
        problem_description = ""
        if title:
            problem_description += f"Title: {title}\n\n"
        if description:
            problem_description += f"{description.strip()}\n\n"
        if input_format:
            problem_description += f"Input Format:\n{input_format.strip()}\n\n"
        if output_format:
            problem_description += f"Output Format:\n{output_format.strip()}\n\n"
        if examples_text:
            problem_description += examples_text
        if note:
            problem_description += f"Note:\n{note.strip()}\n\n"
        
        problem_description = problem_description.strip()
        
        # Extract solution code from accepted_solutions - IMPROVED VERSION
        solution_code = ""
        if 'accepted_solutions' in record and record['accepted_solutions']:
            # Handle the case where accepted_solutions is directly a string with code
            if isinstance(record['accepted_solutions'], str):
                # Just use it directly - no JSON parsing needed
                solution_code = record['accepted_solutions']
            # For completeness, handle the case where it might be a list or dict
            elif isinstance(record['accepted_solutions'], list) and record['accepted_solutions']:
                # Select a random solution if multiple are available
                solution = random.choice(record['accepted_solutions'])
                if isinstance(solution, dict) and 'code' in solution:
                    solution_code = solution['code']
                else:
                    solution_code = str(solution)
            elif isinstance(record['accepted_solutions'], dict) and 'code' in record['accepted_solutions']:
                solution_code = record['accepted_solutions']['code']
        
        # Skip problems without valid solutions
        if not solution_code:
            logger.warning(f"Skipping problem {problem_id} with no valid solution code")
            continue
        
        # Use editorial as explanation
        explanation = editorial if editorial else "No editorial explanation available."

        test_examples = record.get('examples', []) if record.get('examples') else []
        
        processed_data.append({
            "id": problem_id,
            "problem": problem_description,
            "examples": test_examples,
            "solution": solution_code,
            "explanation": explanation
        })
    
    logger.info(f"Successfully processed {len(processed_data)} problems")
    
    # Handle the case where no problems were processed
    if not processed_data:
        logger.error("No problems were processed. Check your dataset format.")
        # Create minimal dummy data to avoid train_test_split error
        dummy_problem = {
            "id": "dummy",
            "problem": "Dummy problem for testing",
            "solution": "def solution():\n    return 'dummy'",
            "explanation": "This is a dummy problem."
        }
        processed_data.append(dummy_problem)
        logger.warning("Added a dummy problem to avoid empty dataset error.")
    
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
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved training data to {train_path}")
    logger.info(f"Saved evaluation data to {eval_path}")
    
    return train_path, eval_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess JSON dataset for DeepCodeRL")
    parser.add_argument("--json_path", type=str, required=True,
                      help="Path to JSON dataset file")
    parser.add_argument("--output_dir", type=str, default="data/processed/", 
                      help="Directory to save processed data")
    parser.add_argument("--train_ratio", type=float, default=0.9, 
                      help="Ratio of data to use for training")
    parser.add_argument("--eval_ratio", type=float, default=0.1, 
                      help="Ratio of data to use for evaluation")
    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed for splitting")
    
    args = parser.parse_args()
    
    preprocess_json_dataset(
        args.json_path,
        args.output_dir,
        args.train_ratio,
        args.eval_ratio,
        args.seed
    )