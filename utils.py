import signal
import sys
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Global flag to track termination request
termination_requested = False

def signal_handler(sig, frame):
    """Handle signals like SIGINT (Ctrl+C) and SIGTERM"""
    global termination_requested
    print("\nGraceful termination requested. Finishing current batch and saving checkpoint...", 
          file=sys.stderr)
    termination_requested = True
    # Set a more aggressive handler for repeated Ctrl+C
    signal.signal(signal.SIGINT, force_exit_handler)

def force_exit_handler(sig, frame):
    """Force exit on second Ctrl+C"""
    print("\nForce terminating now! No checkpoint will be saved.", file=sys.stderr)
    sys.exit(1)

def setup_signal_handlers():
    """Register signal handlers"""
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

def save_checkpoint(model, tokenizer, output_dir, checkpoint_name="interrupted_checkpoint"):
    """Save a checkpoint of the model and tokenizer"""
    checkpoint_path = os.path.join(output_dir, checkpoint_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")