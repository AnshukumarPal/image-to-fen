"""
Chess Board Generator Module

Generates sample chess boards for testing and development.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.fen_generator import FENGenerator

# Chess.com board colors
LIGHT_SQUARE_COLOR = (0xa5, 0xcb, 0xed)  # #edcba5 in BGR
DARK_SQUARE_COLOR = (0x6d, 0xa4, 0xd8)   # #d8a46d in BGR

# Piece representation (emoji for visualization)
PIECE_EMOJI = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
}

class ChessBoardGenerator:
    """Class for generating sample chess boards."""
    
    def __init__(self, square_size=80, board_size=640):
        """
        Initialize the generator.
        
        Args:
            square_size (int): Size of each square in pixels
            board_size (int): Size of the board in pixels
        """
        self.square_size = square_size
        self.board_size = board_size
        self.fen_generator = FENGenerator()
        
        # Ensure board_size is divisible by 8
        if board_size % 8 != 0:
            self.board_size = (board_size // 8) * 8
            print(f"Adjusted board size to {self.board_size} to make it divisible by 8")
        
        # Calculate square size based on board size
        self.square_size = self.board_size // 8
    
    def generate_empty_board(self):
        """
        Generate an empty chess board.
        
        Returns:
            numpy.ndarray: Image of empty chess board
        """
        # Create empty board
        board = np.zeros((self.board_size, self.board_size, 3), dtype=np.uint8)
        
        # Fill with squares
        for row in range(8):
            for col in range(8):
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                # Determine square color
                if (row + col) % 2 == 0:
                    color = LIGHT_SQUARE_COLOR
                else:
                    color = DARK_SQUARE_COLOR
                
                # Draw square
                board[y1:y2, x1:x2] = color
        
        return board
    
    def add_pieces_from_fen(self, board, fen_string):
        """
        Add pieces to a board image based on FEN string.
        
        Args:
            board (numpy.ndarray): Board image
            fen_string (str): FEN notation string
            
        Returns:
            numpy.ndarray: Board image with pieces
        """
        # Make a copy of the board
        board_with_pieces = board.copy()
        
        # Parse FEN position
        position_part = fen_string.split(' ')[0]
        rows = position_part.split('/')
        
        # Process each row
        for row_idx, row in enumerate(rows):
            col_idx = 0
            for char in row:
                if char.isdigit():
                    # Skip empty squares
                    col_idx += int(char)
                else:
                    # Add piece
                    self._draw_piece(board_with_pieces, row_idx, col_idx, char)
                    col_idx += 1
        
        return board_with_pieces
    
    def _draw_piece(self, board, row, col, piece_char):
        """
        Draw a piece on the board.
        
        Args:
            board (numpy.ndarray): Board image
            row (int): Row index (0-7)
            col (int): Column index (0-7)
            piece_char (str): FEN piece character
        """
        # Calculate square coordinates
        x1 = col * self.square_size
        y1 = row * self.square_size
        x2 = x1 + self.square_size
        y2 = y1 + self.square_size
        
        # Draw a placeholder circle for the piece
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = self.square_size // 3
        
        # Choose color based on piece color
        if piece_char.isupper():  # White pieces
            piece_color = (255, 255, 255)
            text_color = (0, 0, 0)
        else:  # Black pieces
            piece_color = (0, 0, 0)
            text_color = (255, 255, 255)
        
        # Draw circle
        cv2.circle(board, (center_x, center_y), radius, piece_color, -1)
        
        # Add piece letter
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(piece_char.upper(), font, 0.8, 2)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2
        cv2.putText(board, piece_char.upper(), (text_x, text_y), font, 0.8, text_color, 2)
    
    def add_move_highlight(self, board, from_square, to_square, highlight_color=(0, 255, 0)):
        """
        Add move highlight to a board image.
        
        Args:
            board (numpy.ndarray): Board image
            from_square (str): From square coordinate (e.g., 'e2')
            to_square (str): To square coordinate (e.g., 'e4')
            highlight_color (tuple): RGB color tuple for highlight
            
        Returns:
            numpy.ndarray: Board image with move highlight
        """
        # Make a copy of the board
        board_with_highlight = board.copy()
        
        # Convert chess notation to indices
        from_col, from_row = ord(from_square[0]) - ord('a'), 8 - int(from_square[1])
        to_col, to_row = ord(to_square[0]) - ord('a'), 8 - int(to_square[1])
        
        # Calculate square coordinates
        from_x1 = from_col * self.square_size
        from_y1 = from_row * self.square_size
        from_x2 = from_x1 + self.square_size
        from_y2 = from_y1 + self.square_size
        
        to_x1 = to_col * self.square_size
        to_y1 = to_row * self.square_size
        to_x2 = to_x1 + self.square_size
        to_y2 = to_y1 + self.square_size
        
        # Draw highlight rectangles
        highlight_thickness = 3
        cv2.rectangle(board_with_highlight, (from_x1, from_y1), (from_x2, from_y2), 
                     highlight_color, highlight_thickness)
        cv2.rectangle(board_with_highlight, (to_x1, to_y1), (to_x2, to_y2), 
                     highlight_color, highlight_thickness)
        
        return board_with_highlight
    
    def add_move_indicator(self, board, square, indicator_type='best', alpha=0.7):
        """
        Add move quality indicator based on Chess.com notation.
        
        Args:
            board (numpy.ndarray): Board image
            square (str): Square coordinate (e.g., 'e4')
            indicator_type (str): Type of indicator: 
                                 'best' (star icon),
                                 'brilliant' (!!),
                                 'great' (!),
                                 'good' (green checkmark),
                                 'inaccuracy' (!?),
                                 'mistake' (?),
                                 'blunder' (??),
                                 'miss' (X)
            alpha (float): Transparency of the indicator
            
        Returns:
            numpy.ndarray: Board image with move indicator
        """
        # Make a copy of the board
        board_with_indicator = board.copy()
        
        # Convert chess notation to indices
        col, row = ord(square[0]) - ord('a'), 8 - int(square[1])
        
        # Calculate square coordinates
        x1 = col * self.square_size
        y1 = row * self.square_size
        x2 = x1 + self.square_size
        y2 = y1 + self.square_size
        
        # Create overlay
        overlay = board_with_indicator.copy()
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = self.square_size // 4
        
        # Choose color and symbol based on indicator type
        if indicator_type == 'best':
            color = (0, 255, 255)  # Yellow
            symbol = '★'
        elif indicator_type == 'brilliant':
            color = (0, 255, 255)  # Yellow
            symbol = '!!'
        elif indicator_type == 'great':
            color = (0, 255, 128)  # Light Green
            symbol = '!'
        elif indicator_type == 'good':
            color = (0, 255, 0)    # Green
            symbol = '✓'
        elif indicator_type == 'inaccuracy':
            color = (0, 165, 255)  # Orange
            symbol = '!?'
        elif indicator_type == 'mistake':
            color = (0, 69, 255)   # Orange-Red
            symbol = '?'
        elif indicator_type == 'blunder':
            color = (0, 0, 255)    # Red
            symbol = '??'
        elif indicator_type == 'miss':
            color = (0, 0, 255)    # Red
            symbol = 'X'
        else:
            color = (200, 200, 200)  # Gray
            symbol = '•'
        
        # Draw indicator circle
        cv2.circle(overlay, (center_x, center_y), radius, color, -1)
        
        # Add symbol
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(symbol, font, 0.8, 2)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2
        cv2.putText(overlay, symbol, (text_x, text_y), font, 0.8, (255, 255, 255), 2)
        
        # Apply overlay with transparency
        cv2.addWeighted(overlay, alpha, board_with_indicator, 1 - alpha, 0, board_with_indicator)
        
        return board_with_indicator
    
    def save_board(self, board, output_path):
        """
        Save a board image to a file.
        
        Args:
            board (numpy.ndarray): Board image
            output_path (str): Path to save the image
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save image
        return cv2.imwrite(output_path, board)
    
    def generate_sample_boards(self, output_dir, num_samples=5):
        """
        Generate sample board images with various positions and indicators.
        
        Args:
            output_dir (str): Directory to save the images
            num_samples (int): Number of sample boards to generate
            
        Returns:
            list: Paths to generated images
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample FEN positions
        sample_positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # After 1.e4
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Ruy Lopez
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R w KQkq - 4 5",  # Italian Game
            "rnbqkb1r/pp2pppp/3p1n2/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 5"  # Sicilian Defense
        ]
        
        # Indicator types and positions
        indicator_samples = [
            ('best', 'e4'),
            ('brilliant', 'd4'),
            ('great', 'c4'),
            ('good', 'd5'),
            ('inaccuracy', 'f6'),
            ('mistake', 'g5'),
            ('blunder', 'h4'),
            ('miss', 'a4')
        ]
        
        # Generate boards
        output_paths = []
        
        for i in range(min(num_samples, len(sample_positions))):
            # Create empty board
            empty_board = self.generate_empty_board()
            
            # Add pieces
            board_with_pieces = self.add_pieces_from_fen(empty_board, sample_positions[i])
            
            # Add highlight for a sample move
            if i < len(indicator_samples):
                # Determine a move for highlight
                if i == 0:
                    from_square, to_square = 'e2', 'e4'
                elif i == 1:
                    from_square, to_square = 'e7', 'e5'
                else:
                    from_square, to_square = 'd2', 'd4'
                
                board_with_highlight = self.add_move_highlight(board_with_pieces, from_square, to_square)
                
                # Add move indicator
                indicator_type, square = indicator_samples[i]
                board_with_indicator = self.add_move_indicator(board_with_highlight, square, indicator_type)
            else:
                board_with_indicator = board_with_pieces
            
            # Save board
            output_path = os.path.join(output_dir, f"sample_board_{i+1}.png")
            self.save_board(board_with_indicator, output_path)
            output_paths.append(output_path)
            
            print(f"Generated sample board: {output_path}")
        
        return output_paths


if __name__ == "__main__":
    # Example usage
    generator = ChessBoardGenerator()
    generator.generate_sample_boards('data/raw', num_samples=5) 