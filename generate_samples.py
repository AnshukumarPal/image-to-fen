#!/usr/bin/env python3
"""
Generate Sample Data

Script to generate sample chess boards and prepare training data.
"""

import os
import sys
from src.chess_board_generator import ChessBoardGenerator
from src.data_preparation import DataPreparation

def main():
    """Main function to generate sample data."""
    # Create data directories
    os.makedirs('data/raw', exist_ok=True)
    
    print("Generating sample chess boards...")
    
    # Generate sample boards
    generator = ChessBoardGenerator()
    board_paths = generator.generate_sample_boards('data/raw', num_samples=5)
    
    print(f"Generated {len(board_paths)} sample boards")
    
    # Ask user if they want to prepare training data
    prepare_data = input("Do you want to prepare training data from the generated boards? (y/n): ")
    
    if prepare_data.lower() == 'y':
        print("Preparing training data...")
        
        # Prepare training data
        data_prep = DataPreparation()
        data_prep.process_directory('data/raw')
        
        # Ask if user wants to augment training data
        augment_data = input("Do you want to augment the training data? (y/n): ")
        
        if augment_data.lower() == 'y':
            print("Augmenting training data...")
            count = data_prep.augment_training_data()
            print(f"Created {count} augmented images")
    
    print("Done!")

if __name__ == "__main__":
    main() 