#!/usr/bin/env python3

# SILENCE ALL WARNINGS - Must be at the very beginning before any imports
import os
# Disable TensorFlow logging completely
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN

# Silence all warnings
import warnings
warnings.filterwarnings('ignore')

# Silence specific TensorFlow and Keras warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR) 

"""
Chess Position Recognition System

Main module that coordinates the workflow of:
1. User-guided cropping
2. Square extraction
3. Piece classification
4. FEN generation with automatic detection of castling rights and en-passant
"""

import argparse
import cv2
import numpy as np

# Import TensorFlow after suppressing warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')

# Suppress deprecation warnings
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from src.board_cropper import BoardCropper
from src.square_extractor import SquareExtractor
from src.piece_classifier import PieceClassifier, FEN_MAPPING
from src.fen_generator import FENGenerator

# Global debug flag
DEBUG = False

def parse_args():
    parser = argparse.ArgumentParser(description='Chess Position Recognition System')
    parser.add_argument('--image', type=str, required=False, help='Path to input image')
    parser.add_argument('--model', type=str, default='models/piece_classifier.h5', 
                        help='Path to trained model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--data-dir', type=str, default='data/training', 
                        help='Directory with training data')
    parser.add_argument('--active-color', type=str, default=None, choices=['w', 'b'],
                        help='Active color for FEN generation (w or b). If not specified, will attempt to detect automatically.')
    parser.add_argument('--en-passant', type=str, default=None, 
                        help='En passant target square in algebraic notation (e.g., e3). If not specified, will attempt to detect automatically.')
    parser.add_argument('--orientation', type=str, default=None, 
                        choices=['white_bottom', 'black_bottom'],
                        help='Force board orientation (default: auto-detect)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    return parser.parse_args()

def detect_active_color_from_image(cropped_board, board_orientation, pieces_matrix=None):
    """
    Detect active color based on highlighted squares indicating the last move.
    Uses rule: the player at the bottom is to move unless highlighted squares 
    indicate they just moved.
    
    Args:
        cropped_board: The cropped chess board image
        board_orientation: 'white_bottom' or 'black_bottom'
        pieces_matrix: Matrix of piece symbols for determining which piece an arrow starts from
        
    Returns:
        Tuple of (active_color, green_arrow_move, highlighted_move)
        active_color: 'w' for white's turn, 'b' for black's turn.
        green_arrow_move: tuple ((from_file, from_rank), (to_file, to_rank)) or None
        highlighted_move: tuple ((from_file, from_rank), (to_file, to_rank)) or None - represents the highlighted previous move
    """
    if cropped_board is None:
        # Can't detect from image, use default
        return 'w' if board_orientation == "white_bottom" else 'b', None, None
    
    # Default: bottom player's turn
    default_color = 'w' if board_orientation == "white_bottom" else 'b'
    green_arrow_move = None
    highlighted_move = None
    
    try:
        # First, check for green arrows (these are special and override other logic)
        # Convert to HSV for better arrow detection
        board_hsv = cv2.cvtColor(cropped_board, cv2.COLOR_BGR2HSV)
        height, width = cropped_board.shape[:2]
        square_size = height // 8
        
        # Green mask specifically for arrows (more permissive than highlight detection)
        # Chess.com uses a bright green for arrows
        # Original range: (40, 100, 100), (85, 255, 255)
        # Wider range to catch more variations:
        arrow_green_mask = cv2.inRange(board_hsv, (35, 50, 50), (95, 255, 255))
        
        # Apply morphology to clean up and enhance arrows
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        arrow_mask = cv2.morphologyEx(arrow_green_mask, cv2.MORPH_CLOSE, kernel)
        
        # Check for significant green arrow presence
        green_pixels = cv2.countNonZero(arrow_mask)
        if DEBUG:
            print(f"Green pixels detected: {green_pixels}, threshold: {square_size * square_size * 0.5}")
        
        if green_pixels > square_size * square_size * 0.5:
            if DEBUG:
                print(f"Detected green arrow with {green_pixels} pixels")
            
            # Find connected components for arrow
            num_arrow_labels, arrow_labels, arrow_stats, _ = cv2.connectedComponentsWithStats(arrow_mask, connectivity=8)
            if DEBUG:
                print(f"Found {num_arrow_labels-1} connected components in green mask")
            
            # Find the largest component (likely the arrow)
            largest_area = 0
            largest_idx = 0
            for i in range(1, num_arrow_labels):
                area = arrow_stats[i, cv2.CC_STAT_AREA]
                if DEBUG:
                    print(f"Component {i}: area={area}, x={arrow_stats[i, cv2.CC_STAT_LEFT]}, y={arrow_stats[i, cv2.CC_STAT_TOP]}")
                if area > largest_area:
                    largest_area = area
                    largest_idx = i
            
            if DEBUG:
                print(f"Largest component: idx={largest_idx}, area={largest_area}")
            
            if largest_idx > 0:
                # Get bounding box of the arrow
                x = arrow_stats[largest_idx, cv2.CC_STAT_LEFT]
                y = arrow_stats[largest_idx, cv2.CC_STAT_TOP]
                w = arrow_stats[largest_idx, cv2.CC_STAT_WIDTH]
                h = arrow_stats[largest_idx, cv2.CC_STAT_HEIGHT]
                
                # Determine start and end squares of the arrow
                # We'll use the extremes of the bounding box
                start_col = min(int(x / square_size), 7)
                start_row = min(int(y / square_size), 7)
                end_col = min(int((x + w - 1) / square_size), 7)
                end_row = min(int((y + h - 1) / square_size), 7)
                
                if DEBUG:
                    print(f"Arrow bounding box: ({x}, {y}, {w}, {h})")
                    print(f"Detected arrow from ({start_row}, {start_col}) to ({end_row}, {end_col})")
                
                # Convert to algebraic notation (a1, e4, etc.)
                from_file = chr(ord('a') + start_col)
                from_rank = 8 - start_row
                to_file = chr(ord('a') + end_col)
                to_rank = 8 - end_row
                
                # If board orientation is black_bottom, we need to flip the coordinates
                if board_orientation == "black_bottom":
                    from_file = chr(ord('a') + (7 - start_col))
                    from_rank = start_row + 1
                    to_file = chr(ord('a') + (7 - end_col))
                    to_rank = end_row + 1
                
                if DEBUG:
                    print(f"Arrow move: {from_file}{from_rank} to {to_file}{to_rank}")
                
                # Store the move for en-passant detection
                green_arrow_move = ((from_file, from_rank), (to_file, to_rank))
                
                # Get piece at the arrow start position
                if pieces_matrix is not None and 0 <= start_row < 8 and 0 <= start_col < 8:
                    # If board orientation is black_bottom, we need to use the flipped coordinates
                    lookup_row = start_row
                    lookup_col = start_col
                    if board_orientation == "black_bottom":
                        lookup_row = 7 - start_row
                        lookup_col = 7 - start_col
                    
                    piece = pieces_matrix[lookup_row][lookup_col]
                    if DEBUG:
                        print(f"Found piece '{piece}' at position [{lookup_row}][{lookup_col}]")
                    
                    # If piece is lowercase, it's black's turn
                    # If piece is uppercase, it's white's turn
                    if piece.islower() and piece != '.':
                        if DEBUG:
                            print(f"Green arrow starts from black piece {piece}")
                        return 'w', green_arrow_move, highlighted_move  # Black just moved, so it's white's turn
                    elif piece.isupper() and piece != '.':
                        if DEBUG:
                            print(f"Green arrow starts from white piece {piece}")
                        return 'b', green_arrow_move, highlighted_move  # White just moved, so it's black's turn
                    else:
                        if DEBUG:
                            print(f"No piece found at arrow start position {lookup_row},{lookup_col}")
                else:
                    if DEBUG:
                        print(f"Arrow start position out of bounds or no pieces matrix")
        
        # Continue with highlight detection if no arrows were found or couldn't determine from arrows
        # Convert to RGB for precise color matching
        board_rgb = cv2.cvtColor(cropped_board, cv2.COLOR_BGR2RGB)
        
        # Define highlight color ranges based on provided RGB values
        # Format: Move feedback - (from square, to square)
        highlight_colors = {
            # Brilliant move - (#7fb388, #89c6a4) - light green
            "brilliant": [(127, 179, 136), (137, 198, 164)],
            # Great move - (#b0b3b2, #a6a096) - gray
            "great": [(176, 179, 178), (166, 160, 150)],
            # Best move - (#b7c078, #adad5c) - olive green
            "best": [(183, 192, 120), (173, 173, 92)],
            # Excellent - (#adad5c, #b7c078) - olive green (reversed)
            "excellent": [(173, 173, 92), (183, 192, 120)],
            # Good - (#b7ae71, #b7ae71) - olive
            "good": [(183, 174, 113), (183, 174, 113)],
            # Inaccuracy - (#f2c86b, #e8b54f) - yellow/gold
            "inaccuracy": [(242, 200, 107), (232, 181, 79)],
            # Mistake - (#f6b77f, #f6b77f) - orange
            "mistake": [(246, 183, 127), (246, 183, 127)],
            # Miss - (#f6a187, #f6a187) - salmon
            "miss": [(246, 161, 135), (246, 161, 135)],
            # Blunder - (#f38669, #e9734d) - red
            "blunder": [(243, 134, 105), (233, 115, 77)]
        }
        
        # Add Chess.com's standard yellow highlight for last move
        # This is a critical addition for proper en-passant detection
        highlight_colors["lastmove"] = [
            (247, 247, 105),  # Brighter yellow
            (242, 221, 94),   # Slightly darker yellow
            (255, 213, 105),  # Amber/gold
            (252, 230, 88)    # Light yellow
        ]
        
        # Create a mask for each set of highlight colors
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Process each highlight color
        for move_type, colors in highlight_colors.items():
            for color in colors:
                # Convert RGB color to numpy array
                target_color = np.array(color, dtype=np.uint8)
                
                # Create mask with a threshold to allow for slight variations
                color_diff = np.abs(board_rgb - target_color)
                color_distance = np.sum(color_diff, axis=2)
                color_mask = (color_distance < 80).astype(np.uint8) * 255
                
                # Add to combined mask
                combined_mask = cv2.bitwise_or(combined_mask, color_mask)
        
        # Try using HSV for yellow detection as an additional method
        # Chess.com often uses yellow for highlighting the last move
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(board_hsv, yellow_lower, yellow_upper)
        combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)
        
        # Apply morphology to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        highlight_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        if DEBUG:
            # Save highlight mask for debugging
            cv2.imwrite('highlight_mask_debug.png', highlight_mask)
            print(f"Saved highlight mask for debugging")
            
            # Count total highlighted pixels
            highlight_pixels = cv2.countNonZero(highlight_mask)
            print(f"Total highlighted pixels: {highlight_pixels}")
        
        # Find all connected components in the highlight mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(highlight_mask, connectivity=8)
        
        # We need to find the highlighted squares (typically 2 - from and to squares)
        highlighted_squares = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Only consider components that are roughly the size of a square
            if square_size * square_size * 0.05 < area < square_size * square_size * 2.0:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Determine which square this is
                square_row = min(int((y + h/2) / square_size), 7)
                square_col = min(int((x + w/2) / square_size), 7)
                
                # Convert to algebraic notation
                file = chr(ord('a') + square_col)
                rank = 8 - square_row
                
                # If board orientation is black_bottom, we need to flip the coordinates
                if board_orientation == "black_bottom":
                    file = chr(ord('a') + (7 - square_col))
                    rank = square_row + 1
                
                # Add to highlighted squares list
                square_info = (file, rank)
                if square_info not in highlighted_squares:
                    highlighted_squares.append(square_info)
                    if DEBUG:
                        print(f"Found highlighted square at {file}{rank}, area={area}, center=({x+w/2}, {y+h/2})")
        
        if DEBUG:
            print(f"Total highlighted squares found: {len(highlighted_squares)}")
        
        # If we found exactly 2 highlighted squares, we can determine the move
        if len(highlighted_squares) == 2:
            # For Chess.com, let's try to determine which square is from and which is to
            # We can use the color intensity or other features to differentiate them
            
            # Sort squares by rank and file to ensure consistency
            highlighted_squares.sort()
            
            # Assume the first square is the "from" square and the second is the "to" square
            # This is a simplification and may not always be correct, but it's a reasonable guess
            from_square, to_square = highlighted_squares
            
            # Create highlighted move tuple
            highlighted_move = (from_square, to_square)
            if DEBUG:
                print(f"Detected highlighted move: {from_square[0]}{from_square[1]} to {to_square[0]}{to_square[1]}")
            
            # Determine the active color based on the pieces
            if pieces_matrix is not None:
                from_file, from_rank = from_square
                to_file, to_rank = to_square
                
                # Get coordinates in the piece matrix
                from_file_idx = ord(from_file) - ord('a')
                from_rank_idx = 8 - int(from_rank)
                to_file_idx = ord(to_file) - ord('a')
                to_rank_idx = 8 - int(to_rank)
                
                # Adjust for board orientation
                if board_orientation == "black_bottom":
                    from_file_idx = 7 - from_file_idx
                    from_rank_idx = int(from_rank) - 1
                    to_file_idx = 7 - to_file_idx
                    to_rank_idx = int(to_rank) - 1
                
                if DEBUG:
                    print(f"From coordinates: ({from_rank_idx}, {from_file_idx}), To coordinates: ({to_rank_idx}, {to_file_idx})")
                
                if 0 <= from_rank_idx < 8 and 0 <= from_file_idx < 8:
                    if DEBUG:
                        try:
                            print(f"Piece matrix at from square: {pieces_matrix[from_rank_idx][from_file_idx]}")
                        except Exception as e:
                            print(f"Error accessing from square in piece matrix: {e}")
                
                if 0 <= to_rank_idx < 8 and 0 <= to_file_idx < 8:
                    try:
                        to_piece = pieces_matrix[to_rank_idx][to_file_idx]
                        if DEBUG:
                            print(f"Piece at highlighted destination: {to_piece}")
                        
                        # Determine active color based on piece
                        if to_piece.islower() and to_piece != '.':
                            # If destination has black piece, it's white's turn
                            return 'w', green_arrow_move, highlighted_move
                        elif to_piece.isupper() and to_piece != '.':
                            # If destination has white piece, it's black's turn
                            return 'b', green_arrow_move, highlighted_move
                    except Exception as e:
                        if DEBUG:
                            print(f"Error determining piece at destination: {e}")
                
                # If we couldn't determine from the piece, use the rank to guess
                # Assuming most moves are from bottom to top for white, top to bottom for black
                if int(from_rank) > int(to_rank):  # Moving up the board
                    # Likely white's move, so now it's black's turn
                    if DEBUG:
                        print(f"Based on direction (upward), assuming white just moved")
                    return 'b', green_arrow_move, highlighted_move
                else:  # Moving down the board
                    # Likely black's move, so now it's white's turn
                    if DEBUG:
                        print(f"Based on direction (downward), assuming black just moved")
                    return 'w', green_arrow_move, highlighted_move
        
        # Analyze for king's position (castling special case)
        # For white_bottom orientation, check e1 and g1 squares for castling
        if board_orientation == "white_bottom":
            # Define e1 and g1 squares
            e1_x, e1_y = 4 * square_size, 7 * square_size
            g1_x, g1_y = 6 * square_size, 7 * square_size
            
            # Check if these squares have highlights
            e1_highlight = cv2.countNonZero(highlight_mask[e1_y:e1_y+square_size, e1_x:e1_x+square_size])
            g1_highlight = cv2.countNonZero(highlight_mask[g1_y:g1_y+square_size, g1_x:g1_x+square_size])
            
            # If both e1 and g1 have highlights, it's likely white castled kingside
            if e1_highlight > 50 and g1_highlight > 50:
                if DEBUG:
                    print("Detected kingside castling (white)")
                # Create a castling move tuple
                highlighted_move = (('e', 1), ('g', 1))
                return 'b', green_arrow_move, highlighted_move  # White just moved, so it's black's turn
        
        # Count bottom vs top highlights
        bottom_half_highlights = 0
        top_half_highlights = 0
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Only consider components that are roughly the size of a square
            if square_size * square_size * 0.05 < area < square_size * square_size * 2:
                # Get component location
                y_center = centroids[i][1]
                
                # Determine if it's in the bottom or top half
                if y_center > height / 2:
                    bottom_half_highlights += 1
                else:
                    top_half_highlights += 1
        
        # Debug output
        if DEBUG:
            print(f"Found {bottom_half_highlights} bottom highlights, {top_half_highlights} top highlights")
        
        # If we found highlights and they're in the bottom half
        if bottom_half_highlights > 0:
            # Bottom player just moved, so it's top player's turn
            if DEBUG:
                print("Highlights found in bottom half - bottom player just moved")
            return 'b' if board_orientation == "white_bottom" else 'w', green_arrow_move, highlighted_move
        
        # For Chess.com's castling, we need a special check
        if board_orientation == "white_bottom":
            # Count pixels in the bottom right corner region (where kingside castling happens)
            bottom_right = highlight_mask[7*square_size:height, 4*square_size:width]
            if cv2.countNonZero(bottom_right) > 50:
                if DEBUG:
                    print("Detected highlights in bottom right (possible castling)")
                return 'b', green_arrow_move, highlighted_move  # White just moved, so it's black's turn
        
        # If we reach here, either no highlights were found or they're mostly in the top half
        # So use the default (bottom player to move)
        return default_color, green_arrow_move, highlighted_move
    except Exception as e:
        # Log error and return default
        if DEBUG:
            print(f"Error in active color detection: {e}")
        return default_color, None, None

def detect_active_color(piece_positions, cropped_board=None, board_orientation=None):
    """
    Detect which side has the next move.
    Simple rule: player at bottom is to move unless highlights show they just moved.
    
    Args:
        piece_positions: List of piece symbols
        cropped_board: Optional cropped board image
        board_orientation: Optional board orientation ('white_bottom' or 'black_bottom')
        
    Returns:
        Tuple of (active_color, green_arrow_move, highlighted_move)
        active_color: 'w' for white's turn, 'b' for black's turn.
        green_arrow_move: Information about detected green arrow for en-passant detection, or None
        highlighted_move: Information about highlighted squares from the previous move, or None
    """
    # Special case for standard starting position
    if is_starting_position(piece_positions):
        return 'w', None, None
    
    # If we have the board image and know orientation, use visual detection
    if cropped_board is not None and board_orientation is not None:
        # Convert piece_positions list to 8x8 matrix
        pieces_matrix = []
        for i in range(8):
            row = []
            for j in range(8):
                index = i * 8 + j
                piece_label = piece_positions[index]
                
                # Convert to FEN symbols
                if piece_label == 'white_pawn':
                    row.append('P')
                elif piece_label == 'white_knight':
                    row.append('N')
                elif piece_label == 'white_bishop':
                    row.append('B')
                elif piece_label == 'white_rook':
                    row.append('R')
                elif piece_label == 'white_queen':
                    row.append('Q')
                elif piece_label == 'white_king':
                    row.append('K')
                elif piece_label == 'black_pawn':
                    row.append('p')
                elif piece_label == 'black_knight':
                    row.append('n')
                elif piece_label == 'black_bishop':
                    row.append('b')
                elif piece_label == 'black_rook':
                    row.append('r')
                elif piece_label == 'black_queen':
                    row.append('q')
                elif piece_label == 'black_king':
                    row.append('k')
                else:
                    row.append('.')
            
            pieces_matrix.append(row)
        
        # Try to detect from image
        active_color, green_arrow_move, highlighted_move = detect_active_color_from_image(cropped_board, board_orientation, pieces_matrix)
        return active_color, green_arrow_move, highlighted_move
    
    # If we only know the orientation but no visual cues, assume bottom player to move
    if board_orientation is not None:
        return 'w' if board_orientation == "white_bottom" else 'b', None, None
    
    # Last resort: default to white's turn
    return 'w', None, None

def is_starting_position(piece_positions):
    """Check if the board shows the standard starting position."""
    
    # Starting position pattern
    starting_position = [
        'black_rook', 'black_knight', 'black_bishop', 'black_queen', 
        'black_king', 'black_bishop', 'black_knight', 'black_rook'
    ]
    starting_position += ['black_pawn'] * 8
    starting_position += ['empty'] * 32
    starting_position += ['white_pawn'] * 8
    starting_position += [
        'white_rook', 'white_knight', 'white_bishop', 'white_queen', 
        'white_king', 'white_bishop', 'white_knight', 'white_rook'
    ]
    
    # Check if positions match
    return piece_positions == starting_position

def main():
    args = parse_args()
    
    # Set debug mode
    global DEBUG
    DEBUG = args.debug
    
    # Create necessary directories
    os.makedirs('data/training', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/debug', exist_ok=True)  # New directory for debug outputs
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
        if DEBUG:
            print(f"Model loaded from {args.model}")
    else:
        print(f"Model not found at {args.model}. Please train the model first.")
        return
    
    # Process the image
    print(f"Processing image: {args.image}")
    
    # Step 1: User-guided cropping
    cropped_board = board_cropper.crop(args.image)
    
    # Save cropped board for debugging
    if DEBUG:
        cv2.imwrite('data/debug/cropped_board.png', cropped_board)
        print("Saved cropped board to data/debug/cropped_board.png")
    
    # Step 2: Extract squares with orientation handling
    squares = square_extractor.extract(cropped_board, orientation=args.orientation)
    
    # Print detected orientation
    print(f"Board orientation: {square_extractor.board_orientation}")
    
    # Step 3: Classify pieces
    piece_positions = piece_classifier.classify_squares(squares)
    
    # Create a visualization of the board with piece positions
    if DEBUG:
        # Create a debug visualization of the board
        board_with_pieces = cropped_board.copy()
        height, width = board_with_pieces.shape[:2]
        square_size = height // 8
        
        # Draw grid lines
        for i in range(9):
            # Horizontal lines
            cv2.line(board_with_pieces, (0, i * square_size), (width, i * square_size), (0, 0, 255), 2)
            # Vertical lines
            cv2.line(board_with_pieces, (i * square_size, 0), (i * square_size, height), (0, 0, 255), 2)
        
        # Add piece labels
        for row in range(8):
            for col in range(8):
                idx = row * 8 + col
                piece = piece_positions[idx]
                if piece != 'empty':
                    piece_label = piece.split('_')[1][0].upper() if 'white' in piece else piece.split('_')[1][0].lower()
                    cv2.putText(board_with_pieces, piece_label, 
                                (col * square_size + 10, row * square_size + square_size - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imwrite('data/debug/board_with_pieces.png', board_with_pieces)
        print("Saved annotated board to data/debug/board_with_pieces.png")
    
    # Step 4: Detect active color if not specified
    active_color, green_arrow_move, highlighted_move = detect_active_color(
        piece_positions, 
        cropped_board=cropped_board, 
        board_orientation=square_extractor.board_orientation
    )
    print(f"Detected active color: {active_color}")
    if green_arrow_move:
        print(f"Detected green arrow move: {green_arrow_move[0][0]}{green_arrow_move[0][1]} to {green_arrow_move[1][0]}{green_arrow_move[1][1]}")
    if highlighted_move:
        print(f"Detected highlighted move: {highlighted_move[0][0]}{highlighted_move[0][1]} to {highlighted_move[1][0]}{highlighted_move[1][1]}")
        
        # Visual debug - highlight the detected move on the board image
        if DEBUG:
            board_with_highlights = cropped_board.copy()
            height, width = board_with_highlights.shape[:2]
            square_size = height // 8
            
            # Highlight the "from" square
            from_file, from_rank = highlighted_move[0]
            from_col = ord(from_file) - ord('a')
            from_row = 8 - int(from_rank)
            if square_extractor.board_orientation == "black_bottom":
                from_col = 7 - from_col
                from_row = int(from_rank) - 1
                
            from_x = from_col * square_size
            from_y = from_row * square_size
            cv2.rectangle(board_with_highlights, (from_x, from_y), 
                         (from_x + square_size, from_y + square_size), (0, 255, 0), 3)
            
            # Highlight the "to" square
            to_file, to_rank = highlighted_move[1]
            to_col = ord(to_file) - ord('a')
            to_row = 8 - int(to_rank)
            if square_extractor.board_orientation == "black_bottom":
                to_col = 7 - to_col
                to_row = int(to_rank) - 1
                
            to_x = to_col * square_size
            to_y = to_row * square_size
            cv2.rectangle(board_with_highlights, (to_x, to_y), 
                         (to_x + square_size, to_y + square_size), (255, 0, 0), 3)
            
            # Draw an arrow from "from" square to "to" square
            from_center_x = from_x + square_size // 2
            from_center_y = from_y + square_size // 2
            to_center_x = to_x + square_size // 2
            to_center_y = to_y + square_size // 2
            
            cv2.arrowedLine(board_with_highlights, (from_center_x, from_center_y), 
                           (to_center_x, to_center_y), (0, 255, 255), 2, cv2.LINE_AA, 0, 0.3)
            
            cv2.imwrite('data/debug/board_with_highlighted_move.png', board_with_highlights)
            print("Saved board with highlighted move to data/debug/board_with_highlighted_move.png")
    
    if DEBUG:
        print("Note: Active color detection is conservative. Use --active-color to set explicitly.")
    
    # Step 5: Generate FEN with detected params
    fen = fen_generator.generate(
        piece_positions, 
        active_color=active_color,
        en_passant=args.en_passant,
        green_arrow_move=green_arrow_move,
        highlighted_move=highlighted_move
    )
    
    print(f"\nFEN Notation: {fen}")
    
    # Save the result for reference
    result_path = os.path.join('data/processed', os.path.basename(args.image))
    board_cropper.save_last_result(result_path)
    if DEBUG:
        print(f"Result saved to {result_path}")

if __name__ == '__main__':
    main() 