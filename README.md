# Chess Position Recognition System

A Python-based system for extracting chess positions from Chess.com screenshots and generating FEN notation.

## Overview

This project provides a complete workflow for recognizing chess positions from Chess.com game screenshots, handling move feedback indicators (brilliant moves, blunders, etc.), and generating FEN notation for further analysis.

### Features

- User-guided cropping of chessboard from screenshots
- Square extraction from the chessboard
- Piece classification with move feedback indicator handling
- FEN notation generation
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

## Project Structure

- `src/`: Source code
  - `board_cropper.py`: User-guided cropping of chessboard
  - `square_extractor.py`: Extraction of 64 squares from board
  - `piece_classifier.py`: Classification of chess pieces
  - `fen_generator.py`: Generation of FEN notation
  - `data_preparation.py`: Tools for preparing training data
  - `chess_board_generator.py`: Generator for sample chess boards
- `data/`: Data directory
  - `raw/`: Raw screenshots
  - `training/`: Training data (organized by piece classes)
  - `processed/`: Processed results
- `models/`: Trained models
- `main.py`: Main application
- `generate_samples.py`: Generate sample data
- `train_model.py`: Train classifier model
- `test_model.py`: Test model on new screenshots

## Implementation Details

### Board Recognition

The system uses a coordinate-based approach to identify the chess board:
1. The user manually crops the chess board from the screenshot
2. The system divides the cropped board into 64 equal squares
3. Each square is classified using a CNN model trained on labeled examples

### Piece Classification

The piece classifier handles Chess.com move feedback indicators:
- Brilliant moves (star icon)
- Good moves (green checkmark)
- Mistakes (red X)
- Blunders (double red X)
- Inaccuracies
- Highlighted squares (previous move)

### FEN Generation

The system generates standard FEN notation including:
- Piece positions
- Active color (assumed white by default)
- Castling availability (assumed none by default)
- En passant target square (assumed none by default)
- Halfmove clock (assumed 0 by default)
- Fullmove number (assumed 1 by default)

## License

[MIT License](LICENSE)
