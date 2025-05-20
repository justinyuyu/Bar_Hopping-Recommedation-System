import argparse
import os
import torch
import pandas as pd
import numpy as np
from .train import train_linear_adapter
from .evaluate import evaluate, plot_loss
from .generate_questions import process_first_n
from .config import get_config
from barhopping.logger import logger

def main():
    parser = argparse.ArgumentParser(description="Bar Hopping Adapter Training and Evaluation")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common argument for config file
    parser.add_argument("--config", type=str, help="Path to config YAML file")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the adapter model")
    train_parser.add_argument("--input-dim", type=int, help="Input dimension of embeddings")
    train_parser.add_argument("--batch-size", type=int, help="Batch size for training")
    train_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, help="Learning rate")
    train_parser.add_argument("--warmup-steps", type=int, help="Number of warmup steps")
    train_parser.add_argument("--margin", type=float, help="Triplet loss margin")
    train_parser.add_argument("--device", type=str, help="Device to use (cpu, cuda, mps)")

    # Generate questions command
    gen_parser = subparsers.add_parser("generate", help="Generate questions for bars")
    gen_parser.add_argument("--num-bars", type=int, help="Number of bars to process")
    gen_parser.add_argument("--questions-per-bar", type=int, help="Questions to generate per bar")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the adapter model")
    eval_parser.add_argument("--k", type=int, help="Top-k for evaluation")
    eval_parser.add_argument("--model-path", type=str, help="Path to saved model")

    args = parser.parse_args()
    
    # Load config
    config = get_config(args.config)
    
    # Override config with command line arguments
    if args.command == "train":
        if args.input_dim: config.input_dim = args.input_dim
        if args.batch_size: config.batch_size = args.batch_size
        if args.epochs: config.epochs = args.epochs
        if args.lr: config.learning_rate = args.lr
        if args.warmup_steps: config.warmup_steps = args.warmup_steps
        if args.margin: config.margin = args.margin
        if args.device: config.device = args.device
        
        logger.info("Starting adapter training...")
        
        """
        # Load your data here
        df = pd.read_csv(os.path.join(config.data_dir, 'train_data.csv'))
        
        adapter, train_losses, val_losses = train_linear_adapter(
            df=df,
            input_dim=config.input_dim,
            batch_size=config.batch_size,
            epochs=config.epochs,
            lr=config.learning_rate,
            warmup_steps=config.warmup_steps,
            margin=config.margin,
            device=config.device
        )
        
        # Save the model
        torch.save(adapter.state_dict(), config.model_save_path)
        logger.info(f"Model saved to {config.model_save_path}")
        plot_loss(train_losses, val_losses)
        """
    
    elif args.command == 'generate':
        if args.num_bars: config.num_bars = args.num_bars
        if args.questions_per_bar: config.questions_per_bar = args.questions_per_bar
        
        logger.info(f"Generating questions for {config.num_bars} bars...")
        process_first_n(n=config.num_bars, num_questions=config.questions_per_bar)
        logger.info("Question generation completed")

    elif args.command == 'evaluate':
        if args.k: config.eval_k = args.k
        if args.model_path: config.model_save_path = args.model_path
        
        logger.info("Evaluating adapter model...")
        
        """
        # Load your test data here
        test_anchors = np.load(os.path.join(config.data_dir, 'test_anchors.npy'))
        test_positives = np.load(os.path.join(config.data_dir, 'test_positives.npy'))
        test_ids = np.load(os.path.join(config.data_dir, 'test_ids.npy'))
        
        mrr, hits = evaluate(test_anchors, test_positives, test_ids, k=config.eval_k)
        logger.info(f"Evaluation results - MRR: {mrr:.4f}, Hits@{config.eval_k}: {hits:.4f}")
        """
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
