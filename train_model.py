#!/usr/bin/env python3
"""
Train Model

Script to train the piece classifier model from labeled data.
"""

import os
import sys
import argparse
from src.piece_classifier import PieceClassifier

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train piece classifier model")
    parser.add_argument('--data-dir', type=str, default='data/training',
                      help='Directory with training data')
    parser.add_argument('--model', type=str, default='models/piece_classifier.h5',
                      help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    return parser.parse_args()

def main():
    """Main function to train the model."""
    args = parse_args()
    
    # Create model directory
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    
    # Check if training data exists
    if not os.path.exists(args.data_dir):
        print(f"Training data directory not found: {args.data_dir}")
        print("Please run 'python generate_samples.py' first to generate and label training data.")
        return
    
    # Check if data directory has class subdirectories
    class_dirs = [d for d in os.listdir(args.data_dir) 
                 if os.path.isdir(os.path.join(args.data_dir, d))]
    
    if not class_dirs:
        print(f"No class directories found in {args.data_dir}")
        print("Please make sure the training data is organized in class subdirectories.")
        return
    
    print(f"Found {len(class_dirs)} class directories: {', '.join(class_dirs)}")
    
    # Initialize classifier
    classifier = PieceClassifier()
    
    # Train model
    print(f"Training model with {args.epochs} epochs and batch size {args.batch_size}...")
    model = classifier.train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
    
    # Save model
    classifier.save_model(args.model)
    print(f"Model saved to {args.model}")

if __name__ == "__main__":
    main() 