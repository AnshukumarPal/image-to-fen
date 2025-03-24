"""
Piece Classifier Module

Classifies chess pieces from square images, handling move feedback indicators.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Define piece labels
PIECE_LABELS = {
    0: 'empty',
    1: 'white_pawn', 2: 'white_knight', 3: 'white_bishop', 
    4: 'white_rook', 5: 'white_queen', 6: 'white_king',
    7: 'black_pawn', 8: 'black_knight', 9: 'black_bishop', 
    10: 'black_rook', 11: 'black_queen', 12: 'black_king'
}

# Define FEN characters for each piece
FEN_MAPPING = {
    'empty': '.',
    'white_pawn': 'P', 'white_knight': 'N', 'white_bishop': 'B',
    'white_rook': 'R', 'white_queen': 'Q', 'white_king': 'K',
    'black_pawn': 'p', 'black_knight': 'n', 'black_bishop': 'b',
    'black_rook': 'r', 'black_queen': 'q', 'black_king': 'k'
}

class PieceClassifier:
    """Class for classifying chess pieces from square images."""
    
    def __init__(self, img_size=(64, 64)):
        """
        Initialize the classifier.
        
        Args:
            img_size (tuple): Size to resize images to (height, width)
        """
        self.img_size = img_size
        self.model = None
        self.class_indices = PIECE_LABELS
    
    def _create_model(self, num_classes=13):
        """
        Create a CNN model for piece classification.
        
        Args:
            num_classes (int): Number of classes to predict
            
        Returns:
            keras.models.Sequential: The compiled model
        """
        model = Sequential([
            # First convolutional layer
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            MaxPooling2D((2, 2)),
            
            # Second convolutional layer
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            # Third convolutional layer
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _preprocess_image(self, image):
        """
        Preprocess a single image for the model.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Resize image
        resized = cv2.resize(image, self.img_size)
        
        # Scale pixel values
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def train(self, data_dir, epochs=20, batch_size=32, validation_split=0.2):
        """
        Train the model on labeled square images.
        
        Args:
            data_dir (str): Directory with training data organized in subdirectories by class
            epochs (int): Number of epochs to train
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            keras.models.Sequential: The trained model
        """
        print(f"Training model on data from {data_dir}")
        
        # Check if data directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Create data generator with augmentation
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # Don't flip chess pieces
            validation_split=validation_split
        )
        
        # Load training data
        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Load validation data
        validation_generator = datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # Update class indices mapping
        self.class_indices = {v: k for k, v in train_generator.class_indices.items()}
        
        # Create model
        self.model = self._create_model(num_classes=len(train_generator.class_indices))
        
        # Train model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size
        )
        
        # Plot training history
        self._plot_training_history(history)
        
        return self.model
    
    def _plot_training_history(self, history):
        """Plot training and validation accuracy/loss."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='lower right')
        
        # Plot loss
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_path):
        """
        Save the trained model.
        
        Args:
            model_path (str): Path to save the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.model is None:
            print("No model to save. Train a model first.")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        
        # Save class indices
        class_indices_path = os.path.join(os.path.dirname(model_path), 'class_indices.npy')
        np.save(class_indices_path, self.class_indices)
        
        print(f"Model saved to {model_path}")
        print(f"Class indices saved to {class_indices_path}")
        
        return True
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return False
        
        # Load model
        self.model = load_model(model_path)
        
        # Load class indices if available
        class_indices_path = os.path.join(os.path.dirname(model_path), 'class_indices.npy')
        if os.path.exists(class_indices_path):
            self.class_indices = np.load(class_indices_path, allow_pickle=True).item()
        
        print(f"Model loaded from {model_path}")
        
        return True
    
    def classify_square(self, square_img):
        """
        Classify a single square image.
        
        Args:
            square_img (numpy.ndarray): Square image to classify
            
        Returns:
            str: Piece label (e.g., 'white_pawn', 'black_queen', 'empty')
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        # Preprocess image
        preprocessed = self._preprocess_image(square_img)
        
        # Predict
        prediction = self.model.predict(preprocessed, verbose=0)
        class_idx = np.argmax(prediction[0])
        
        # Map index to label
        return self.class_indices[class_idx]
    
    def classify_squares(self, squares):
        """
        Classify multiple square images.
        
        Args:
            squares (list): List of square images
            
        Returns:
            list: List of piece labels in the same order
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        # Process each square
        results = []
        for square in squares:
            label = self.classify_square(square)
            results.append(label)
        
        return results 