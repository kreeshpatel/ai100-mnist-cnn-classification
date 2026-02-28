# MNIST Handwritten Digit Classification with CNN

AI 100 Midterm Project — Deep learning classification of handwritten digits using a Convolutional Neural Network (CNN) implemented in PyTorch on the MNIST dataset.

## Project Overview

This project trains a CNN to classify 28x28 grayscale images of handwritten digits (0–9) from the MNIST dataset. The model achieves high accuracy using a simple two-layer convolutional architecture followed by fully connected layers.

### Model Architecture

```
Input (1x28x28)
  → Conv2d(1, 32, 3, padding=1) → ReLU → MaxPool(2)    -> 32x14x14
  → Conv2d(32, 64, 3, padding=1) → ReLU → MaxPool(2)   -> 64x7x7
  → Flatten                                              -> 3136
  → Linear(3136, 128) → ReLU
  → Linear(128, 10)                                      -> 10 logits
```

## Setup

### Prerequisites

- Python 3.8 or higher

### Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train the model (default settings)

```bash
python src/train.py
```

### Train with custom hyperparameters

```bash
python src/train.py --epochs 10 --batch_size 128 --lr 0.0005 --seed 42
```

### Command-line Arguments

| Argument       | Default | Description                      |
|----------------|---------|----------------------------------|
| `--epochs`     | 5       | Number of training epochs        |
| `--batch_size` | 64      | Batch size for train and test    |
| `--lr`         | 0.001   | Learning rate for Adam optimizer |
| `--seed`       | 42      | Random seed for reproducibility  |

## Outputs

After training completes, all results are saved to the `outputs/` folder:

| File                    | Description                          |
|-------------------------|--------------------------------------|
| `metrics.json`          | Training losses, accuracies, config  |
| `loss_curve.png`        | Training loss over epochs            |
| `acc_curve.png`         | Training accuracy over epochs        |
| `confusion_matrix.png`  | Confusion matrix on the test set     |

## Project Structure

```
ai100-mnist-cnn-classification/
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── model.py              # CNN model definition
│   ├── train.py              # Training and evaluation script
│   └── utils.py              # Helper functions for metrics and plots
└── outputs/
    └── .gitkeep              # Placeholder (outputs saved here after training)
```

## Dependencies

- **torch** — Deep learning framework
- **torchvision** — MNIST dataset and image transforms
- **matplotlib** — Plotting loss/accuracy curves
- **scikit-learn** — Confusion matrix generation
- **numpy** — Numerical utilities
