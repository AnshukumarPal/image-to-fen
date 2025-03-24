"""
FEN Generator Module

Converts a chess position to FEN (Forsyth-Edwards Notation).
"""

from src.piece_classifier import FEN_MAPPING

class FENGenerator:
    """Class for generating FEN notation from piece positions."""
    
    def __init__(self):
        self.piece_positions = []
        self.last_fen = None
    
    def generate(self, piece_positions):
        """
        Generate FEN notation from piece positions.
        
        Args:
            piece_positions (list): List of 64 piece labels in row-major order (A8-H8, A7-H7, ..., A1-H1)
            
        Returns:
            str: FEN notation string
        """
        if len(piece_positions) != 64:
            raise ValueError(f"Expected 64 piece positions, got {len(piece_positions)}")
        
        self.piece_positions = piece_positions
        
        # Start with an empty FEN string
        fen_parts = []
        
        # Process each row
        for row in range(8):
            row_start = row * 8
            row_end = row_start + 8
            row_pieces = piece_positions[row_start:row_end]
            
            # Convert row to FEN
            row_fen = self._row_to_fen(row_pieces)
            fen_parts.append(row_fen)
        
        # Join rows with '/'
        position_part = '/'.join(fen_parts)
        
        # For simplicity, assume it's white's turn, no castling rights, no en passant
        full_fen = f"{position_part} w - - 0 1"
        self.last_fen = full_fen
        
        return full_fen
    
    def _row_to_fen(self, row_pieces):
        """
        Convert a row of pieces to FEN notation.
        
        Args:
            row_pieces (list): List of 8 piece labels for a row
            
        Returns:
            str: FEN notation for the row
        """
        # Convert each piece to its FEN character
        fen_chars = [FEN_MAPPING.get(piece, '.') for piece in row_pieces]
        
        # Compress empty squares
        compressed = []
        empty_count = 0
        
        for char in fen_chars:
            if char == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    compressed.append(str(empty_count))
                    empty_count = 0
                compressed.append(char)
        
        # Append any remaining empty squares
        if empty_count > 0:
            compressed.append(str(empty_count))
        
        # Join characters to form row FEN
        return ''.join(compressed)
    
    def parse_fen(self, fen_string):
        """
        Parse a FEN string to piece positions.
        
        Args:
            fen_string (str): FEN notation string
            
        Returns:
            list: List of 64 piece labels
        """
        # Split FEN into parts
        parts = fen_string.strip().split(' ')
        position_part = parts[0]
        
        # Split position into rows
        rows = position_part.split('/')
        
        if len(rows) != 8:
            raise ValueError(f"Invalid FEN string: expected 8 rows, got {len(rows)}")
        
        # Create empty piece positions list
        piece_positions = []
        
        # Process each row
        for row in rows:
            row_pieces = []
            for char in row:
                if char.isdigit():
                    # Add empty squares
                    empty_count = int(char)
                    row_pieces.extend(['empty'] * empty_count)
                else:
                    # Add piece
                    for piece_name, fen_char in FEN_MAPPING.items():
                        if fen_char == char:
                            row_pieces.append(piece_name)
                            break
            
            # Add row to piece positions
            piece_positions.extend(row_pieces)
        
        self.piece_positions = piece_positions
        return piece_positions
    
    def visualize_fen(self, fen_string=None):
        """
        Visualize a FEN string as a text board.
        
        Args:
            fen_string (str): FEN notation string. If None, use the last generated FEN.
            
        Returns:
            str: Text representation of the board
        """
        if fen_string is None:
            if self.last_fen is None:
                raise ValueError("No FEN string provided and no previous FEN available")
            fen_string = self.last_fen
        
        # Parse the FEN
        parts = fen_string.strip().split(' ')
        position_part = parts[0]
        
        # Split position into rows
        rows = position_part.split('/')
        
        # Create text board
        board_text = []
        board_text.append('  +------------------------+')
        
        for i, row in enumerate(rows):
            rank = 8 - i
            row_text = f"{rank} |"
            col = 0
            
            for char in row:
                if char.isdigit():
                    # Add empty squares
                    empty_count = int(char)
                    row_text += ' .' * empty_count
                    col += empty_count
                else:
                    # Add piece
                    row_text += f" {char}"
                    col += 1
            
            row_text += " |"
            board_text.append(row_text)
        
        board_text.append('  +------------------------+')
        board_text.append('    a b c d e f g h')
        
        # Join and return
        return '\n'.join(board_text)
    
    def get_piece_count(self, fen_string=None):
        """
        Count pieces in a FEN string.
        
        Args:
            fen_string (str): FEN notation string. If None, use the last generated FEN.
            
        Returns:
            dict: Dictionary with piece counts
        """
        if fen_string is None:
            if self.last_fen is None:
                raise ValueError("No FEN string provided and no previous FEN available")
            fen_string = self.last_fen
        
        # Parse the FEN
        parts = fen_string.strip().split(' ')
        position_part = parts[0]
        
        # Count pieces
        piece_counts = {
            'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'K': 0,
            'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0, 'k': 0
        }
        
        for char in position_part:
            if char in piece_counts:
                piece_counts[char] += 1
        
        return piece_counts 