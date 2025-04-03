# Chess Position Recognition System

A Python-based system for extracting chess positions from Chess.com screenshots and generating FEN notation.

## Overview

This project provides a complete workflow for recognizing chess positions from Chess.com game screenshots, handling move feedback indicators (brilliant moves, blunders, etc.), and generating FEN notation for further analysis.

### Features

- User-guided cropping of chessboard from screenshots
- Advanced board orientation detection using multiple techniques:
  - Square color pattern analysis
  - Piece color detection
  - Highlighted move detection
- Piece classification with move feedback indicator handling
- Intelligent FEN generation with:
  - Automatic castling rights detection based on piece positions
  - Heuristic-based en-passant detection
  - Visual active color detection based on highlighted moves
- Training data preparation tools
- Model training and testing tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chess-analysis.git
cd chess-analysis
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generating Sample Data

To generate sample chess boards and prepare training data:

```bash
python generate_samples.py
```

This script will:
1. Generate sample chess boards in `data/raw/`
2. Guide you through preparing the training data by labeling squares
3. Optionally augment the training data to improve model accuracy

### Training the Model

To train the piece classifier model:

```bash
python train_model.py
```

Options:
- `--data-dir`: Directory with training data (default: `data/training`)
- `--model`: Path to save trained model (default: `models/piece_classifier.h5`)
- `--epochs`: Number of epochs to train (default: `20`)
- `--batch-size`: Batch size for training (default: `32`)

### Testing the Model

To test the trained model on a new chess screenshot:

```bash
python test_model.py --image path/to/screenshot.png
```

Options:
- `--image`: Path to input image (required)
- `--model`: Path to trained model (default: `models/piece_classifier.h5`)
- `--output`: Directory to save results (default: `data/processed`)
- `--visualize`: Visualize the results (flag)

### Using the Main Application

To process a chess screenshot:

```bash
python main.py --image path/to/screenshot.png
```

Options:
- `--image`: Path to input image
- `--model`: Path to trained model (default: `models/piece_classifier.h5`)
- `--train`: Train the model (flag)
- `--data-dir`: Directory with training data (default: `data/training`)
- `--active-color`: Active color for FEN generation (w or b) (default: auto-detect)
- `--en-passant`: En passant target square in algebraic notation (e.g., e3) (default: auto-detect)
- `--orientation`: Force board orientation ('white_bottom' or 'black_bottom', default: auto-detect)

## Project Structure

- `src/`: Source code
  - `board_cropper.py`: User-guided cropping of chessboard
  - `square_extractor.py`: Extraction of 64 squares from board with advanced orientation detection
  - `piece_classifier.py`: Classification of chess pieces
  - `fen_generator.py`: Generation of FEN notation with castling and en-passant detection
  - `data_preparation.py`: Tools for preparing training data
  - `chess_board_generator.py`: Generator for sample chess boards
- `data/`: Data directory
  - `raw/`: Raw screenshots
  - `training/`: Training data (organized by piece classes)
  - `processed/`: Processed results
- `models/`: Trained models
- `main.py`: Main application with visual feature detection
- `generate_samples.py`: Generate sample data
- `train_model.py`: Train classifier model
- `test_model.py`: Test model on new screenshots

## Implementation Details

### Board Recognition

The system uses a coordinate-based approach to identify the chess board:
1. The user manually crops the chess board from the screenshot
2. The system automatically detects board orientation through multiple methods:
   - Square color pattern analysis (bottom-right should be light)
   - Piece color detection on bottom rank
   - Highlighted move detection for Chess.com screenshots
3. The system divides the cropped board into 64 equal squares
4. Each square is classified using a CNN model trained on labeled examples

### Piece Classification

The piece classifier handles Chess.com move feedback indicators:
- Best move (star icon)
- Brilliant moves (!!)
- Great moves (!)
- Good moves (green checkmark)
- Inaccuracies (!?)
- Mistakes (?)
- Blunders (??)
- Miss (X)

### FEN Generation

The system generates standard FEN notation including:
- Piece positions
- Active color (detected from highlighted squares showing last move)
- Castling availability (detected based on king and rook positions)
- En passant target square (detected by analyzing pawn positions)
- Halfmove clock (assumed 0 by default)
- Fullmove number (assumed 1 by default)

### Automatic Detection Features

- **Board Orientation**: Uses multiple methods including square colors, piece colors, and highlighted moves to determine orientation
- **Active Color**: Primarily detects whose turn it is based on highlighted moves in Chess.com screenshots
- **En-passant**: Identifies potential en-passant situations by analyzing pawn positions on ranks 4 and 5
- **Castling Rights**: Determines castling possibilities by checking if kings and rooks are in their original positions

## License

[MIT License](LICENSE)
