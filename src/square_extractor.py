"""
Square Extractor Module

Extracts the 64 squares from a cropped chessboard image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.piece_classifier import PieceClassifier

class SquareExtractor:
    """Class for extracting chess squares from a board image."""
    
    def __init__(self, square_size=None):
        """
        Initialize the square extractor.
        
        Args:
            square_size (int, optional): Size of each square. If None, it will be calculated.
        """
        self.square_size = square_size
        self.squares = []
        self.board_orientation = "white_bottom"  # "white_bottom" or "black_bottom"
    
    def extract(self, board_image, orientation=None):
        """
        Extract the 64 squares from a board image.
        
        Args:
            board_image (numpy.ndarray): Cropped chess board image
            orientation (str, optional): Force a specific orientation ("white_bottom" or "black_bottom")
            
        Returns:
            list: List of 64 square images in row-major order (A8-H8, A7-H7, ..., A1-H1)
        """
        if board_image is None:
            raise ValueError("No board image provided")
        
        # Store a copy of the original image
        self.board_image = board_image.copy()
        
        # Get board dimensions
        height, width = board_image.shape[:2]
        
        # Calculate square size
        if self.square_size is None:
            self.square_size = min(height, width) // 8
        
        # Detect board orientation if not specified
        if orientation is not None:
            self.board_orientation = orientation
        else:
            self.board_orientation = self.detect_orientation(board_image)
        
        # Extract squares
        self.squares = []
        for row in range(8):
            for col in range(8):
                # Calculate square coordinates
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                # Make sure we don't go out of bounds
                x2 = min(x2, width)
                y2 = min(y2, height)
                
                # Extract square
                square = board_image[y1:y2, x1:x2]
                
                # Append to list
                self.squares.append(square)
        
        # Flip the squares order if the board orientation is black at bottom
        if self.board_orientation == "black_bottom":
            # Reverse the entire array to flip the board orientation
            self.squares = self.squares[::-1]
        
        return self.squares
    
    def detect_orientation(self, board_image):
        """
        Detect the board orientation based on piece placement.
        1. Check kings and queens at the bottom rank
        2. Check piece colors at the bottom rank
        
        Args:
            board_image (numpy.ndarray): Cropped chess board image
            
        Returns:
            str: "white_bottom" or "black_bottom"
        """
        height, width = board_image.shape[:2]
        square_size = min(height, width) // 8
        
        # Extract all squares to analyze pieces
        self.piece_classifier = PieceClassifier()
        self.piece_classifier.load_model("models/piece_classifier.h5")
        squares = self.extract_squares(board_image)
        
        # Method 1: Check kings and queens at bottom rank (most reliable)
        bottom_rank = []
        for col in range(8):
            square = board_image[(7*square_size):height, col*square_size:(col+1)*square_size]
            bottom_rank.append(square)
            
        piece_labels = []
        for square in bottom_rank:
            piece_labels.append(self.piece_classifier.classify_square(square))
        
        # Look for king and queen positions
        # In standard chess notation, white king starts at e1, queen at d1
        if 'white_king' in piece_labels and 'white_queen' in piece_labels:
            k_index = piece_labels.index('white_king')
            q_index = piece_labels.index('white_queen')
            # Check if king and queen are in e1 and d1 positions
            if k_index == 4 and q_index == 3:
                return "white_bottom"
        
        # Check for black king and queen at bottom
        if 'black_king' in piece_labels and 'black_queen' in piece_labels:
            k_index = piece_labels.index('black_king')
            q_index = piece_labels.index('black_queen')
            # Check if king and queen are in e1 and d1 positions
            if k_index == 4 and q_index == 3:
                return "black_bottom"
        
        # Method 2: Check piece colors at the bottom rank
        white_pieces = sum(1 for p in piece_labels if p.startswith('white_') and p != 'empty')
        black_pieces = sum(1 for p in piece_labels if p.startswith('black_') and p != 'empty')
        
        if white_pieces > black_pieces:
            return "white_bottom"
        elif black_pieces > white_pieces:
            return "black_bottom"
        
        # Method 3: Fallback to square color check
        h1_x = 7 * square_size
        h1_y = 7 * square_size
        h1_square = board_image[h1_y:h1_y+square_size, h1_x:h1_x+square_size]
        
        a1_x = 0
        a1_y = 7 * square_size
        a1_square = board_image[a1_y:a1_y+square_size, a1_x:a1_x+square_size]
        
        # Convert squares to grayscale
        h1_gray = cv2.cvtColor(h1_square, cv2.COLOR_BGR2GRAY) if len(h1_square.shape) > 2 else h1_square
        a1_gray = cv2.cvtColor(a1_square, cv2.COLOR_BGR2GRAY) if len(a1_square.shape) > 2 else a1_square
        
        # Calculate average brightness
        h1_brightness = np.mean(h1_gray)
        a1_brightness = np.mean(a1_gray)
        
        # In standard chess, h1 should be light and a1 should be dark
        # If this pattern is reversed, the board is likely flipped
        return "black_bottom" if h1_brightness < a1_brightness else "white_bottom"
    
    def extract_squares(self, board_image):
        """Helper method to extract all squares without applying orientation logic"""
        height, width = board_image.shape[:2]
        square_size = min(height, width) // 8
        
        squares = []
        for row in range(8):
            for col in range(8):
                x1 = col * square_size
                y1 = row * square_size
                x2 = min(x1 + square_size, width)
                y2 = min(y1 + square_size, height)
                square = board_image[y1:y2, x1:x2]
                squares.append(square)
        
        return squares
    
    def visualize_squares(self):
        """
        Visualize the extracted squares.
        
        Returns:
            numpy.ndarray: Visualization image
        """
        if not self.squares:
            raise ValueError("No squares to visualize. Extract squares first.")
        
        # Create a grid of squares
        rows = []
        for i in range(8):
            row_start = i * 8
            row_end = row_start + 8
            row_squares = self.squares[row_start:row_end]
            
            # Concatenate squares horizontally
            row = np.hstack(row_squares)
            rows.append(row)
        
        # Concatenate rows vertically
        visualization = np.vstack(rows)
        
        return visualization
    
    def save_squares(self, output_dir):
        """
        Save the extracted squares to files.
        
        Args:
            output_dir (str): Directory to save the squares
            
        Returns:
            list: Paths to saved square images
        """
        if not self.squares:
            raise ValueError("No squares to save. Extract squares first.")
        
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define square coordinates
        files = 'abcdefgh'
        ranks = '87654321'
        
        # Save each square
        square_paths = []
        for i, square in enumerate(self.squares):
            file_idx = i % 8
            rank_idx = i // 8
            
            # Get coordinates
            square_name = f"{files[file_idx]}{ranks[rank_idx]}"
            
            # Save square
            square_path = os.path.join(output_dir, f"square_{square_name}.png")
            cv2.imwrite(square_path, square)
            square_paths.append(square_path)
        
        return square_paths 