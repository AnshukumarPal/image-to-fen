#!/usr/bin/env python3
"""
Chess Position Recognition System

Main module that coordinates the workflow of:
1. User-guided cropping
2. Square extraction
3. Piece classification
4. FEN generation
"""

import os
import argparse
from src.board_cropper import BoardCropper
from src.square_extractor import SquareExtractor
from src.piece_classifier import PieceClassifier
from src.fen_generator import FENGenerator

def parse_args():
    parser = argparse.ArgumentParser(description='Chess Position Recognition System')
    parser.add_argument('--image', type=str, required=False, help='Path to input image')
    parser.add_argument('--model', type=str, default='models/piece_classifier.h5', 
                        help='Path to trained model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--data-dir', type=str, default='data/training', 
                        help='Directory with training data')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create necessary directories
    os.makedirs('data/training', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    if args.train:
        # Train the model
        classifier = PieceClassifier()
        classifier.train(args.data_dir)
        classifier.save_model(args.model)
        print(f"Model trained and saved to {args.model}")
        return
    
    if not args.image:
        print("Please provide an image path using --image")
        return
    
    # Initialize components
    board_cropper = BoardCropper()
    square_extractor = SquareExtractor()
    piece_classifier = PieceClassifier()
    fen_generator = FENGenerator()
    
    # Load pre-trained model
    if os.path.exists(args.model):
        piece_classifier.load_model(args.model)
    else:
        print(f"Model not found at {args.model}. Please train the model first.")
        return
    
    # Process the image
    print(f"Processing image: {args.image}")
    
    # Step 1: User-guided cropping
    cropped_board = board_cropper.crop(args.image)
    
    # Step 2: Extract squares
    squares = square_extractor.extract(cropped_board)
    
    # Step 3: Classify pieces
    piece_positions = piece_classifier.classify_squares(squares)
    
    # Step 4: Generate FEN
    fen = fen_generator.generate(piece_positions)
    
    print(f"\nFEN Notation: {fen}")
    
    # Save the result for reference
    result_path = os.path.join('data/processed', os.path.basename(args.image))
    board_cropper.save_last_result(result_path)
    print(f"Result saved to {result_path}")

if __name__ == '__main__':
    main() 