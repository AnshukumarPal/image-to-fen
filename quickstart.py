#!/usr/bin/env python3
"""
Quickstart Guide

Interactive script to guide users through the entire workflow:
1. Generate sample data
2. Prepare training data
3. Train model
4. Test model
"""

import os
import sys
import subprocess
import time

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50 + "\n")

def wait_for_enter(message="Press Enter to continue..."):
    """Wait for user to press Enter."""
    input(message)

def main():
    """Main function to guide user through the workflow."""
    print_section("Chess Position Recognition System - Quickstart Guide")
    
    print("This guide will walk you through the process of:")
    print("1. Generating sample chess boards")
    print("2. Preparing training data")
    print("3. Training the model")
    print("4. Testing the model on a chess position")
    
    wait_for_enter()
    
    # Step 1: Generate sample boards
    print_section("Step 1: Generate Sample Chess Boards")
    
    print("This step will generate sample chess boards for training.")
    print("These boards will have chess pieces and move feedback indicators.")
    
    proceed = input("Proceed with generating sample boards? (y/n): ")
    
    if proceed.lower() == 'y':
        print("\nRunning generate_samples.py...")
        subprocess.run([sys.executable, "generate_samples.py"])
    else:
        print("\nSkipping sample generation. Make sure you have data in data/raw/")
    
    wait_for_enter()
    
    # Step 2: Train the model
    print_section("Step 2: Train the Model")
    
    print("This step will train the piece classifier on your labeled data.")
    print("The training process may take a while depending on your system.")
    
    proceed = input("Proceed with training the model? (y/n): ")
    
    if proceed.lower() == 'y':
        print("\nRunning train_model.py...")
        subprocess.run([sys.executable, "train_model.py"])
    else:
        print("\nSkipping model training. Make sure you have a trained model in models/piece_classifier.h5")
    
    wait_for_enter()
    
    # Step 3: Test the model
    print_section("Step 3: Test the Model")
    
    print("This step will test the trained model on a chess position.")
    print("You will be guided to select a chess board from a screenshot.")
    
    # First check if there are any sample boards to test on
    sample_boards = [f for f in os.listdir("data/raw") 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if sample_boards:
        test_image = os.path.join("data/raw", sample_boards[0])
        print(f"\nFound sample image: {test_image}")
    else:
        test_image = input("\nEnter path to a chess screenshot (or press Enter to skip): ")
    
    if test_image:
        print("\nRunning test_model.py...")
        subprocess.run([sys.executable, "test_model.py", "--image", test_image, "--visualize"])
    else:
        print("\nSkipping model testing.")
    
    wait_for_enter()
    
    # Final step: Instructions for real-world use
    print_section("Instructions for Real-World Use")
    
    print("Now that you've completed the quickstart guide, you can:")
    print("1. Use main.py to process chess screenshots:")
    print("   python main.py --image path/to/screenshot.png")
    print("\n2. Collect more training data:")
    print("   - Add more screenshots to data/raw/")
    print("   - Run generate_samples.py to label them")
    print("\n3. Improve the model:")
    print("   - Collect more diverse training examples")
    print("   - Retrain with train_model.py")
    
    print("\nThank you for using the Chess Position Recognition System!")

if __name__ == "__main__":
    main() 