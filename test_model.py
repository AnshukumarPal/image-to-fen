#!/usr/bin/env python3
"""
Test Model

Script to test the trained piece classifier on new chess screenshots.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
from src.board_cropper import BoardCropper
from src.square_extractor import SquareExtractor
from src.piece_classifier import PieceClassifier
from src.fen_generator import FENGenerator

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test piece classifier on chess screenshots")
    parser.add_argument('--image', type=str, required=True,
                      help='Path to input image')
    parser.add_argument('--model', type=str, default='models/piece_classifier.h5',
                      help='Path to trained model')
    parser.add_argument('--output', type=str, default='data/processed',
                      help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize the results')
    return parser.parse_args()

def main():
    """Main function to test the model."""
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Please run 'python train_model.py' first to train the model.")
        return
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize components
    board_cropper = BoardCropper()
    square_extractor = SquareExtractor()
    piece_classifier = PieceClassifier()
    fen_generator = FENGenerator()
    
    # Load model
    print(f"Loading model from {args.model}...")
    piece_classifier.load_model(args.model)
    
    # Process image
    print(f"Processing image: {args.image}")
    
    # Step 1: User-guided cropping
    print("Step 1: Cropping chess board (select area by dragging a rectangle)")
    cropped_board = board_cropper.crop(args.image)
    
    # Step 2: Extract squares
    print("Step 2: Extracting squares")
    squares = square_extractor.extract(cropped_board)
    
    # Visualize squares if requested
    if args.visualize:
        square_extractor.visualize_squares()
    
    # Step 3: Classify pieces
    print("Step 3: Classifying squares")
    piece_positions = piece_classifier.classify_squares(squares)
    
    # Step 4: Generate FEN
    print("Step 4: Generating FEN notation")
    fen = fen_generator.generate(piece_positions)
    
    print(f"\nFEN Notation: {fen}")
    
    # Display text representation of the board
    board_text = fen_generator.visualize_fen(fen)
    print("\nBoard:")
    print(board_text)
    
    # Count pieces
    piece_counts = fen_generator.get_piece_count(fen)
    print("\nPiece counts:")
    for piece, count in piece_counts.items():
        if count > 0:
            print(f"{piece}: {count}")
    
    # Save the result
    output_path = os.path.join(args.output, os.path.basename(args.image))
    board_cropper.save_last_result(output_path)
    print(f"\nResult saved to {output_path}")

if __name__ == "__main__":
    main() 