"""
Square Extractor Module

Extracts individual squares from a chess board image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

class SquareExtractor:
    """Class for extracting individual squares from a chess board image."""
    
    def __init__(self):
        self.board_img = None
        self.squares = []
        self.last_board_img = None
    
    def extract(self, board_img):
        """
        Extract 64 individual squares from a chess board image.
        
        Args:
            board_img (numpy.ndarray): Chess board image
            
        Returns:
            list: List of 64 square images in row-major order (A8-H8, A7-H7, ..., A1-H1)
        """
        self.board_img = board_img
        self.last_board_img = board_img.copy()
        height, width = board_img.shape[:2]
        
        # Calculate square size
        square_height = height // 8
        square_width = width // 8
        
        # Extract squares
        self.squares = []
        for row in range(8):
            for col in range(8):
                # Extract square
                y1 = row * square_height
                y2 = (row + 1) * square_height
                x1 = col * square_width
                x2 = (col + 1) * square_width
                
                square = board_img[y1:y2, x1:x2]
                self.squares.append(square)
        
        return self.squares
    
    def visualize_squares(self):
        """
        Visualize the extracted squares.
        """
        if not self.squares:
            print("No squares extracted yet.")
            return
        
        # Create a figure to display all 64 squares
        fig, axes = plt.subplots(8, 8, figsize=(12, 12))
        
        # Display each square
        for idx, square in enumerate(self.squares):
            row, col = divmod(idx, 8)
            rgb_square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(rgb_square)
            axes[row, col].axis('off')
            
            # Add coordinate labels
            file_label = chr(ord('a') + col)
            rank_label = 8 - row
            axes[row, col].set_title(f"{file_label}{rank_label}", fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
    def adjust_extraction(self, board_img, num_rows=8, num_cols=8):
        """
        Adjust the extraction process with specific number of rows/cols.
        Useful for non-standard boards or testing.
        
        Args:
            board_img (numpy.ndarray): Chess board image
            num_rows (int): Number of rows
            num_cols (int): Number of columns
            
        Returns:
            list: List of extracted square images
        """
        self.board_img = board_img
        self.last_board_img = board_img.copy()
        height, width = board_img.shape[:2]
        
        # Calculate square size
        square_height = height // num_rows
        square_width = width // num_cols
        
        # Extract squares
        self.squares = []
        for row in range(num_rows):
            for col in range(num_cols):
                # Extract square
                y1 = row * square_height
                y2 = (row + 1) * square_height
                x1 = col * square_width
                x2 = (col + 1) * square_width
                
                square = board_img[y1:y2, x1:x2]
                self.squares.append(square)
        
        return self.squares 