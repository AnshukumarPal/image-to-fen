"""
FEN Generator Module

Converts a chess position to FEN (Forsyth-Edwards Notation).
Features enhanced castling rights detection based on piece positions.
"""

from src.piece_classifier import FEN_MAPPING

class FENGenerator:
    """
    Class for generating FEN notation from piece positions.
    
    Features:
    - Automatic castling rights detection based on king and rook positions
    - Customizable active color
    - En-passant target square detection
    - Standard FEN notation generation
    """
    
    def __init__(self):
        self.piece_positions = []
        self.last_fen = None
    
    def generate(self, piece_positions, active_color='w', en_passant=None, green_arrow_move=None, highlighted_move=None):
        """
        Generate FEN notation from piece positions.
        
        Args:
            piece_positions (list): List of 64 piece labels in row-major order (A8-H8, A7-H7, ..., A1-H1)
            active_color (str): Active color ('w' or 'b')
            en_passant (str, optional): En passant target square in algebraic notation (e.g., 'e3')
                If None, attempts to detect possible en-passant targets.
            green_arrow_move (tuple, optional): From and to coordinates of a green arrow detected in the image
                Format: ((from_file, from_rank), (to_file, to_rank))
            highlighted_move (tuple, optional): From and to coordinates of the highlighted previous move
                Format: ((from_file, from_rank), (to_file, to_rank))
            
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
        
        # Determine castling rights
        castling_rights = self._determine_castling_rights(piece_positions)
        
        # Determine en passant target square if not provided
        if en_passant is None:
            en_passant = self.detect_possible_en_passant(piece_positions, active_color, green_arrow_move, highlighted_move)
        
        # Complete FEN string
        full_fen = f"{position_part} {active_color} {castling_rights} {en_passant} 0 1"
        self.last_fen = full_fen
        
        return full_fen
    
    def _determine_castling_rights(self, piece_positions):
        """
        Determine castling rights from piece positions.
        
        Args:
            piece_positions (list): List of 64 piece labels
            
        Returns:
            str: Castling rights part of FEN notation
        """
        # Check if kings and rooks are in their original positions
        white_king_at_e1 = piece_positions[60] == 'white_king'
        black_king_at_e8 = piece_positions[4] == 'black_king'
        
        white_kingside_rook = piece_positions[63] == 'white_rook'
        white_queenside_rook = piece_positions[56] == 'white_rook'
        black_kingside_rook = piece_positions[7] == 'black_rook'
        black_queenside_rook = piece_positions[0] == 'black_rook'
        
        # Determine castling rights
        castling = ''
        if white_king_at_e1:
            if white_kingside_rook:
                castling += 'K'
            if white_queenside_rook:
                castling += 'Q'
        
        if black_king_at_e8:
            if black_kingside_rook:
                castling += 'k'
            if black_queenside_rook:
                castling += 'q'
        
        # If no castling rights, return '-'
        return castling if castling else '-'
    
    def detect_possible_en_passant(self, piece_positions, active_color, green_arrow_move=None, highlighted_move=None):
        """
        Detect possible en-passant target squares based on pawn positions, green arrows, and highlighted moves.
        
        Args:
            piece_positions (list): List of 64 piece labels
            active_color (str): Active color ('w' or 'b')
            green_arrow_move (tuple, optional): From and to coordinates of a green arrow 
                Format: ((from_file, from_rank), (to_file, to_rank))
            highlighted_move (tuple, optional): From and to coordinates of the highlighted previous move
                Format: ((from_file, from_rank), (to_file, to_rank))
            
        Returns:
            str: En passant target square in algebraic notation, or '-' if none
        """
        # Define helper function to convert index to algebraic notation
        def index_to_algebraic(idx):
            file_idx = idx % 8
            rank_idx = idx // 8
            file_char = chr(ord('a') + file_idx)
            rank_char = str(8 - rank_idx)
            return file_char + rank_char
        
        # Get piece name at a specific algebraic coordinate
        def get_piece_at(file, rank):
            file_idx = ord(file) - ord('a')
            rank_idx = 8 - int(rank)
            idx = rank_idx * 8 + file_idx
            if 0 <= idx < 64:
                return piece_positions[idx]
            return None
        
        # Print board state for debugging
        def print_board_debug():
            print("Current board state:")
            for i in range(8):
                row = []
                for j in range(8):
                    idx = i * 8 + j
                    piece = piece_positions[idx]
                    if piece == 'empty':
                        row.append('.')
                    elif piece == 'white_pawn':
                        row.append('P')
                    elif piece == 'black_pawn':
                        row.append('p')
                    elif piece == 'white_king':
                        row.append('K')
                    elif piece == 'black_king':
                        row.append('k')
                    else:
                        row.append(piece[0].upper() if 'white' in piece else piece[0].lower())
                print(''.join(row))
        
        # Debug info
        print("Detecting en-passant opportunities...")
        print(f"Active color: {active_color}")
        if highlighted_move:
            from_square, to_square = highlighted_move
            print(f"Highlighted move: {from_square[0]}{from_square[1]} to {to_square[0]}{to_square[1]}")
            
            # For Chess.com, we need to correct the move direction
            # The highlighted squares are often detected in the wrong order
            # Check if the pieces make sense for the detected move
            from_file, from_rank = from_square
            to_file, to_rank = to_square
            
            # Convert ranks to integers for comparison
            from_rank = int(from_rank)
            to_rank = int(to_rank)
            
            # Get pieces at the positions
            from_piece = get_piece_at(from_file, from_rank)
            to_piece = get_piece_at(to_file, to_rank)
            
            print(f"Initial detection - From piece: {from_piece}, To piece: {to_piece}")
            
            # Check if the move makes sense
            if from_piece == 'empty' and to_piece is not None and to_piece != 'empty':
                # The move is likely in reverse - swap from and to
                print(f"Swapping move direction: {to_file}{to_rank} -> {from_file}{from_rank}")
                highlighted_move = ((to_file, to_rank), (from_file, from_rank))
                from_square, to_square = highlighted_move
                from_file, from_rank = from_square
                to_file, to_rank = to_square
                from_rank = int(from_rank)
                to_rank = int(to_rank)
                from_piece = get_piece_at(from_file, from_rank) 
                to_piece = get_piece_at(to_file, to_rank)
                print(f"Corrected move: {from_file}{from_rank} -> {to_file}{to_rank}")
                print(f"Corrected - From piece: {from_piece}, To piece: {to_piece}")
            
            # Additional Chess.com specific correction for the common case of pawn moves
            # If we have a pawn at one location and empty at the other, correct the direction
            if from_piece == 'black_pawn' and to_piece == 'empty' and from_rank < to_rank:
                # Black pawns move down the board (rank decreases)
                print(f"Correcting black pawn move direction: {to_file}{to_rank} -> {from_file}{from_rank}")
                highlighted_move = ((to_file, to_rank), (from_file, from_rank))
                from_square, to_square = highlighted_move
                from_file, from_rank = from_square
                to_file, to_rank = to_square
                from_rank = int(from_rank)
                to_rank = int(to_rank)
                from_piece = get_piece_at(from_file, from_rank)
                to_piece = get_piece_at(to_file, to_rank)
            elif from_piece == 'white_pawn' and to_piece == 'empty' and from_rank > to_rank:
                # White pawns move up the board (rank increases)
                print(f"Correcting white pawn move direction: {to_file}{to_rank} -> {from_file}{from_rank}")
                highlighted_move = ((to_file, to_rank), (from_file, from_rank))
                from_square, to_square = highlighted_move
                from_file, from_rank = from_square
                to_file, to_rank = to_square
                from_rank = int(from_rank)
                to_rank = int(to_rank)
                from_piece = get_piece_at(from_file, from_rank)
                to_piece = get_piece_at(to_file, to_rank)
        
        # First, check for highlighted moves (most reliable indicator)
        if highlighted_move is not None:
            from_square, to_square = highlighted_move
            from_file, from_rank = from_square
            to_file, to_rank = to_square
            
            # Convert ranks to integers for comparison
            from_rank = int(from_rank)
            to_rank = int(to_rank)
            
            print(f"Analyzing highlighted move: {from_file}{from_rank} -> {to_file}{to_rank}")
            
            # Check if this looks like a pawn move
            from_piece = get_piece_at(from_file, from_rank)
            to_piece = get_piece_at(to_file, to_rank)
            
            print(f"From piece: {from_piece}, To piece: {to_piece}")
            
            # Critical case: Check if this is a BLACK pawn move from rank 7 to 5
            if (active_color == 'w' and  # If it's white's turn, black just moved
                from_piece == 'black_pawn' and  # From piece is a black pawn
                from_rank == 7 and to_rank == 5 and  # Moved 2 squares
                from_file == to_file):  # Same file (straight move)
                # This creates an en-passant opportunity on the middle square
                print(f"En-passant opportunity detected: Black pawn moved from {from_file}{from_rank} to {to_file}{to_rank}")
                en_passant_square = from_file + '6'  # The middle square
                print(f"Setting en-passant target to {en_passant_square}")
                return en_passant_square
            
            # Check for white pawn creating en-passant
            if (active_color == 'b' and  # If it's black's turn, white just moved
                from_piece == 'white_pawn' and  # From piece is a white pawn
                from_rank == 2 and to_rank == 4 and  # Moved 2 squares
                from_file == to_file):  # Same file (straight move)
                # This creates an en-passant opportunity on the middle square
                print(f"En-passant opportunity detected: White pawn moved from {from_file}{from_rank} to {to_file}{to_rank}")
                en_passant_square = from_file + '3'  # The middle square
                print(f"Setting en-passant target to {en_passant_square}")
                return en_passant_square
            
            # Check based on piece positions if we can't identify the pieces correctly
            if to_piece == 'black_pawn' and abs(from_rank - to_rank) == 2:
                # If to location has a black pawn and it moved 2 squares, it came from rank 7
                print(f"Detected black pawn move by position: {from_file}{from_rank} -> {to_file}{to_rank}")
                en_passant_square = to_file + '6'
                print(f"Setting en-passant target to {en_passant_square}")
                return en_passant_square
            elif to_piece == 'white_pawn' and abs(from_rank - to_rank) == 2:
                # If to location has a white pawn and it moved 2 squares, it came from rank 2
                print(f"Detected white pawn move by position: {from_file}{from_rank} -> {to_file}{to_rank}")
                en_passant_square = to_file + '3'
                print(f"Setting en-passant target to {en_passant_square}")
                return en_passant_square
        
        # Second, check for green arrows (also reliable)
        if green_arrow_move is not None:
            from_pos, to_pos = green_arrow_move
            from_file, from_rank = from_pos
            to_file, to_rank = to_pos
            
            # Convert ranks to integers if they're strings
            if isinstance(from_rank, str):
                from_rank = int(from_rank)
            if isinstance(to_rank, str):
                to_rank = int(to_rank)
                
            print(f"Analyzing green arrow: {from_file}{from_rank} -> {to_file}{to_rank}")
            
            # Get the piece at the to position
            to_piece = get_piece_at(to_file, to_rank)
            
            # If a pawn just moved two squares
            if to_piece == 'white_pawn' and from_rank == 2 and to_rank == 4:
                # The en-passant square is between the from and to positions
                return to_file + '3'
            elif to_piece == 'black_pawn' and from_rank == 7 and to_rank == 5:
                # The en-passant square is between the from and to positions
                return to_file + '6'
        
        # Third, check by examining the current position
        
        # Check for potential en-passant with white to move
        if active_color == 'w':
            # Look for black pawns on the 5th rank adjacent to white pawns
            for rank_idx in range(3, 4):  # 4th rank (index 3)
                for file_idx in range(8):
                    idx = rank_idx * 8 + file_idx
                    if idx < len(piece_positions) and piece_positions[idx] == 'black_pawn':
                        # Check if this black pawn is on the 5th rank
                        # and likely just moved two squares (from 7th rank)
                        
                        # Check if there are adjacent white pawns that could capture en-passant
                        # Left white pawn
                        if file_idx > 0 and piece_positions[(rank_idx+1)*8 + file_idx-1] == 'white_pawn':
                            print(f"Detected potential en-passant: Black pawn at {chr(ord('a')+file_idx)}{8-rank_idx} with white pawn to the left")
                            return chr(ord('a') + file_idx) + '6'
                        
                        # Right white pawn
                        if file_idx < 7 and piece_positions[(rank_idx+1)*8 + file_idx+1] == 'white_pawn':
                            print(f"Detected potential en-passant: Black pawn at {chr(ord('a')+file_idx)}{8-rank_idx} with white pawn to the right")
                            return chr(ord('a') + file_idx) + '6'
        
        # Check for potential en-passant with black to move
        elif active_color == 'b':
            # Look for white pawns on the 4th rank adjacent to black pawns
            for rank_idx in range(4, 5):  # 5th rank (index 4)
                for file_idx in range(8):
                    idx = rank_idx * 8 + file_idx
                    if idx < len(piece_positions) and piece_positions[idx] == 'white_pawn':
                        # Check if this white pawn is on the 4th rank
                        # and likely just moved two squares (from 2nd rank)
                        
                        # Check if there are adjacent black pawns that could capture en-passant
                        # Left black pawn
                        if file_idx > 0 and piece_positions[(rank_idx-1)*8 + file_idx-1] == 'black_pawn':
                            print(f"Detected potential en-passant: White pawn at {chr(ord('a')+file_idx)}{8-rank_idx} with black pawn to the left")
                            return chr(ord('a') + file_idx) + '3'
                        
                        # Right black pawn
                        if file_idx < 7 and piece_positions[(rank_idx-1)*8 + file_idx+1] == 'black_pawn':
                            print(f"Detected potential en-passant: White pawn at {chr(ord('a')+file_idx)}{8-rank_idx} with black pawn to the right")
                            return chr(ord('a') + file_idx) + '3'
        
        # No en-passant found
        print("No en-passant opportunities detected")
        return '-'
    
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