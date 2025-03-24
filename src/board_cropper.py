"""
Board Cropper Module

Provides functionality for user-guided cropping of chessboard from screenshots.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

class BoardCropper:
    """Class for handling user-guided cropping of chessboard images."""
    
    def __init__(self):
        self.cropped_image = None
        self.original_image = None
        self.fig = None
        self.ax = None
        self.coords = None
    
    def onselect(self, eclick, erelease):
        """Callback for rectangle selection."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.coords = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        
    def crop(self, image_path):
        """
        Allows user to crop a chessboard from an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Cropped image of the chessboard
        """
        # Read image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        
        # Convert from BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Set up the interactive plot
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.imshow(image_rgb)
        self.ax.set_title('Select chessboard area (drag to create rectangle, close window when done)')
        
        # Set up the RectangleSelector
        rs = RectangleSelector(
            self.ax, self.onselect, useblit=True,
            button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
        )
        
        plt.connect('key_press_event', lambda event: self._on_key(event, rs))
        
        # Display the plot to allow user selection
        plt.tight_layout()
        plt.show()
        
        # Crop the image if coordinates were selected
        if self.coords:
            x, y, w, h = self.coords
            self.cropped_image = self.original_image[y:y+h, x:x+w]
            
            # Show the cropped result
            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2RGB))
            plt.title('Cropped Chessboard')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            return self.cropped_image
        else:
            print("No area selected, returning original image")
            self.cropped_image = self.original_image
            return self.original_image
    
    def _on_key(self, event, rect_selector):
        """Handle key press events for the selector."""
        if event.key == 'enter':
            plt.close()
    
    def save_last_result(self, output_path):
        """Save the last cropped image to a file."""
        if self.cropped_image is not None:
            cv2.imwrite(output_path, self.cropped_image)
            return True
        return False 