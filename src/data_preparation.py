"""
Data Preparation Module

Utilities for preparing and organizing training data.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from src.board_cropper import BoardCropper
from src.square_extractor import SquareExtractor
from src.fen_generator import FENGenerator
from tqdm import tqdm

class DataPreparation:
    """Class for preparing and organizing training data."""
    
    def __init__(self, data_root='data'):
        """
        Initialize the data preparation utility.
        
        Args:
            data_root (str): Root directory for data
        """
        self.data_root = data_root
        self.training_dir = os.path.join(data_root, 'training')
        self.raw_dir = os.path.join(data_root, 'raw')
        
        # Create necessary directories
        os.makedirs(self.training_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)
        
        # Initialize components
        self.board_cropper = BoardCropper()
        self.square_extractor = SquareExtractor()
        self.fen_generator = FENGenerator()
        
        # Initialize piece classes
        self.piece_classes = [
            'empty',
            'white_pawn', 'white_knight', 'white_bishop', 
            'white_rook', 'white_queen', 'white_king',
            'black_pawn', 'black_knight', 'black_bishop', 
            'black_rook', 'black_queen', 'black_king'
        ]
        
        # Create class directories
        for piece_class in self.piece_classes:
            os.makedirs(os.path.join(self.training_dir, piece_class), exist_ok=True)
    
    def process_screenshot(self, image_path):
        """
        Process a chess.com screenshot to extract and label squares.
        
        Args:
            image_path (str): Path to the screenshot
            
        Returns:
            list: Paths to saved square images
        """
        print(f"Processing image: {image_path}")
        
        # Crop board
        cropped_board = self.board_cropper.crop(image_path)
        
        # Extract squares
        squares = self.square_extractor.extract(cropped_board)
        
        # Visualize squares for labeling
        self._label_squares(squares, os.path.basename(image_path))
    
    def _label_squares(self, squares, image_name):
        """
        Interactive labeling of squares.
        
        Args:
            squares (list): List of square images
            image_name (str): Base name of the source image
        """
        piece_counts = {label: 0 for label in self.piece_classes}
        
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(bottom=0.2)  # Reduce bottom margin to make more room for buttons
        square_idx = [0]  # Use list for nonlocal access
        
        # Display initial square
        img = cv2.cvtColor(squares[square_idx[0]], cv2.COLOR_BGR2RGB)
        img_display = ax.imshow(img)
        title = ax.set_title(f"Square {square_idx[0]+1}/64 - Select piece type")
        
        # Create buttons for each piece type
        button_axes = []
        buttons = []
        
        # Calculate button positions - UPDATED FOR BETTER LAYOUT
        button_height = 0.04  # Smaller height
        button_width = 0.13   # Slightly smaller width
        button_spacing_x = 0.01  # Less spacing
        button_spacing_y = 0.01  # Less spacing
        
        # Organize piece classes into groups with more columns
        grouped_classes = [
            ['empty'],
            ['white_pawn', 'white_knight', 'white_bishop', 'white_rook', 'white_queen', 'white_king'],
            ['black_pawn', 'black_knight', 'black_bishop', 'black_rook', 'black_queen', 'black_king']
        ]
        
        # Create buttons for each piece class - IMPROVED LAYOUT
        for i, group in enumerate(grouped_classes):
            if i == 0:  # Special case for 'empty'
                # Place empty button centered at top
                button_ax = plt.axes([0.4, 0.17, button_width, button_height])
                button = Button(button_ax, 'empty')
                
                # Define button callback
                def make_callback(pc):
                    def callback(event):
                        nonlocal piece_counts
                        self._save_square(squares[square_idx[0]], pc, image_name, piece_counts[pc])
                        piece_counts[pc] += 1
                        
                        # Move to next square
                        square_idx[0] += 1
                        if square_idx[0] < len(squares):
                            # Update image
                            img = cv2.cvtColor(squares[square_idx[0]], cv2.COLOR_BGR2RGB)
                            img_display.set_data(img)
                            title.set_text(f"Square {square_idx[0]+1}/64 - Select piece type")
                            plt.draw()
                        else:
                            # Finished all squares
                            plt.close(fig)
                            print("All squares labeled!")
                            print(f"Piece counts: {piece_counts}")
                    return callback
                
                button.on_clicked(make_callback('empty'))
                button_axes.append(button_ax)
                buttons.append(button)
            else:
                # Create a more compact 6-column layout for white and black pieces
                for j, piece_class in enumerate(group):
                    # Calculate button position with tighter spacing
                    left = 0.1 + (j % 6) * (button_width + button_spacing_x)
                    bottom = 0.17 - (i + (j // 6)) * (button_height + button_spacing_y)
                    
                    # Create button
                    button_ax = plt.axes([left, bottom, button_width, button_height])
                    button = Button(button_ax, piece_class)
                    
                    button.on_clicked(make_callback(piece_class))
                    button_axes.append(button_ax)
                    buttons.append(button)
        
        # Add skip button
        skip_ax = plt.axes([0.4, 0.05, 0.2, 0.04])
        skip_button = Button(skip_ax, 'Skip this square')
        
        def skip_callback(event):
            # Move to next square without saving
            square_idx[0] += 1
            if square_idx[0] < len(squares):
                # Update image
                img = cv2.cvtColor(squares[square_idx[0]], cv2.COLOR_BGR2RGB)
                img_display.set_data(img)
                title.set_text(f"Square {square_idx[0]+1}/64 - Select piece type")
                plt.draw()
            else:
                # Finished all squares
                plt.close(fig)
                print("All squares labeled!")
                print(f"Piece counts: {piece_counts}")
        
        skip_button.on_clicked(skip_callback)
        
        plt.show()
    
    def _save_square(self, square_img, piece_class, image_name, count):
        """
        Save a labeled square to the appropriate class directory.
        
        Args:
            square_img (numpy.ndarray): Square image
            piece_class (str): Piece class label
            image_name (str): Base name of source image
            count (int): Counter for this piece type
            
        Returns:
            str: Path to saved image
        """
        # Create filename
        filename = f"{piece_class}_{image_name}_{count}.png"
        save_path = os.path.join(self.training_dir, piece_class, filename)
        
        # Save image
        cv2.imwrite(save_path, square_img)
        return save_path
    
    def process_directory(self, image_dir):
        """
        Process all chess.com screenshots in a directory.
        
        Args:
            image_dir (str): Directory containing screenshots
            
        Returns:
            int: Number of images processed
        """
        print(f"Processing images in directory: {image_dir}")
        
        # Get list of image files
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Process each image
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(image_dir, image_file)
            self.process_screenshot(image_path)
        
        return len(image_files)
    
    def augment_training_data(self, rotation_range=15, brightness_range=(0.8, 1.2)):
        """
        Augment training data with rotations and brightness adjustments.
        
        Args:
            rotation_range (int): Maximum rotation angle in degrees
            brightness_range (tuple): Range of brightness adjustment factors
            
        Returns:
            int: Number of augmented images created
        """
        print("Augmenting training data...")
        
        count = 0
        
        # Process each class directory
        for piece_class in self.piece_classes:
            class_dir = os.path.join(self.training_dir, piece_class)
            
            # Skip if directory doesn't exist
            if not os.path.exists(class_dir):
                continue
            
            # Get list of image files
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Skip if no images found
            if not image_files:
                continue
            
            print(f"Augmenting {len(image_files)} images for class: {piece_class}")
            
            # Process each image
            for image_file in tqdm(image_files, desc=f"Augmenting {piece_class}"):
                image_path = os.path.join(class_dir, image_file)
                
                # Read image
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                # Generate augmented images
                for i in range(5):  # Create 5 augmented versions of each image
                    # Random rotation
                    angle = np.random.uniform(-rotation_range, rotation_range)
                    h, w = img.shape[:2]
                    center = (w // 2, h // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(img, rotation_matrix, (w, h), 
                                           borderMode=cv2.BORDER_REFLECT)
                    
                    # Random brightness
                    brightness = np.random.uniform(*brightness_range)
                    adjusted = cv2.convertScaleAbs(rotated, alpha=brightness, beta=0)
                    
                    # Save augmented image
                    aug_filename = f"{os.path.splitext(image_file)[0]}_aug{i}.png"
                    aug_path = os.path.join(class_dir, aug_filename)
                    cv2.imwrite(aug_path, adjusted)
                    count += 1
        
        print(f"Created {count} augmented images")
        return count


if __name__ == "__main__":
    # Example usage
    data_prep = DataPreparation()
    
    # Process a directory of screenshots
    # data_prep.process_directory('data/raw')
    
    # Augment training data
    # data_prep.augment_training_data() 