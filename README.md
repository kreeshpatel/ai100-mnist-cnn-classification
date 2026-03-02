# MNIST Handwritten Digit Classification with CNNs

**AI 100 Midterm Project** -- Deep learning classification of handwritten digits (0-9) using a Convolutional Neural Network built with PyTorch.

| Metric | Value |
|--------|-------|
| Test Accuracy | **98.84%** |
| Parameters | ~206K |
| Training Time | ~1 min (CPU) |
| Framework | PyTorch |

## Project Structure

```
ai100-mnist-cnn-classification/
├── src/
│   ├── model.py          # CNN architecture (MNISTNet)
│   ├── train.py           # Training and evaluation script
│   ├── utils.py           # Metrics and plotting utilities
│   └── sanity_check.py    # Environment and dependency checker
├── outputs/               # Training artifacts (plots, metrics, report)
├── data/                  # MNIST dataset (auto-downloaded)
├── requirements.txt
└── README.md
```

## Setup

**Prerequisites:** Python 3.10+ (3.11 recommended)

```bash
# Clone the repository
git clone https://github.com/kreeshpatel/ai100-mnist-cnn-classification.git
cd ai100-mnist-cnn-classification

# Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Verify your environment

```bash
python src/sanity_check.py
```

This checks Python version, all dependencies (including PyTorch), CUDA availability, model instantiation, and runs a mini training loop. If anything fails it prints actionable fix instructions.

### 2. Train the model

```bash
python src/train.py
```

Default hyperparameters:

| Parameter | Default |
|-----------|---------|
| Epochs | 3 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Seed | 42 |

You can override any of them:

```bash
python src/train.py --epochs 5 --batch_size 128 --lr 0.0005 --seed 123
```

Training outputs are saved to `outputs/`:
- `metrics.json` -- loss/accuracy per epoch and final test accuracy
- `loss_curve.png` -- training loss over epochs
- `acc_curve.png` -- training accuracy over epochs
- `confusion_matrix.png` -- 10x10 confusion matrix on the test set

## Model Architecture

```
Input (1 x 28 x 28)
  -> Conv2d(1, 32, 3, padding=1) -> ReLU -> MaxPool(2)    [32 x 14 x 14]
  -> Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool(2)   [64 x  7 x  7]
  -> Flatten                                                [3136]
  -> Linear(3136, 128) -> ReLU
  -> Linear(128, 10)                                        [10 logits]
```

Two convolutional blocks extract spatial features (edges, curves, digit shapes), each followed by ReLU activation and 2x2 max-pooling. The flattened features pass through two fully connected layers to produce class logits. CrossEntropyLoss handles the softmax internally.

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) is automatically downloaded on first run to `data/`.

- **Training set:** 60,000 grayscale 28x28 images
- **Test set:** 10,000 grayscale 28x28 images
- **Classes:** 10 (digits 0-9)
- **Preprocessing:** Normalize with mean=0.1307, std=0.3081

## Results

| Epoch | Training Loss | Training Accuracy |
|-------|--------------|-------------------|
| 1 | 0.2317 | 93.32% |
| 2 | 0.0568 | 98.29% |
| 3 | 0.0386 | 98.80% |

**Final Test Accuracy: 98.84%**

The model converges quickly -- above 98% training accuracy by epoch 2 with minimal overfitting.

## Troubleshooting

If `import torch` fails on Windows, run the sanity check first (`python src/sanity_check.py`) -- it prints a detailed checklist covering the Visual C++ Redistributable, 64-bit Python, and correct PyTorch wheel installation.

## Author

**Kreesh Patel** -- AI 100 Midterm Project
