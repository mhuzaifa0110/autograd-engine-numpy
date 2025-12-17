# Automatic Differentiation Engine from Scratch

A complete implementation of automatic differentiation (autograd) engine using only NumPy, and training neural networks with it.

## Project Overview

This project implements:
1. **Automatic Differentiation Engine** - Reverse-mode AD with full NumPy compatibility
2. **Neural Network Training** - MLP with 3 hidden layers (128 neurons each)
3. **Real Dataset Training** - MNIST / Fashion-MNIST / CIFAR-10 (grayscale)
4. **Plots & Analysis** - Loss/accuracy curves, LR schedule, overfitting gap, Hessian utilities

## Restrictions

- ✅ **Allowed:** Python stdlib, NumPy, Matplotlib, scikit-learn (data loading only)
- ❌ **Forbidden:** PyTorch, TensorFlow, JAX, Autograd, TinyGrad, MyGrad, or any third-party autograd code

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

**Option A: Using main.py (Recommended)**
```bash
# Train (you will be prompted to select dataset)
python main.py train

# Train with explicit dataset
python main.py train --dataset mnist
python main.py train --dataset fashion
python main.py train --dataset cifar10

# Compare ReLU vs Tanh on a dataset
python main.py compare --dataset mnist --epochs 30
```

**Option B: Using train.py directly**
```bash
python train.py
```

This will:
- Train a model (or both activations if you call it that way)
- Generate and save plots to `plots/`
- Save results JSON into `results/`
- Save the trained model into `models/`

### 3. Make Predictions

**Using main.py:**
```bash
# Uses the latest model in models/ automatically
python main.py predict --show-examples 10

# Or specify a model explicitly
python main.py predict --model models/model_relu_mnist.pkl --show-examples 10
```

**Or using Python directly:**
```python
from predict import load_model, predict
from train import load_mnist

model = load_model('models/model_relu_mnist.pkl')
_, X_test, _, y_test = load_mnist()
predictions, probs = predict(model, X_test)
```

### 4. Run Analysis

**Using main.py:**
```bash
# Performance analysis
python main.py analyze --type performance

# Scaling analysis
python main.py analyze --type scaling

# Hessian eigenvalues
python main.py analyze --type hessian --model models/model_relu_mnist.pkl
```

**Or using analysis.py directly:**
```bash
python analysis.py
```

## File Structure

```
Project/
├── main.py          # Main entry point (recommended)
├── autograd.py      # Core Tensor class with autograd
├── nn.py            # Neural network layers
├── optim.py         # SGD optimizer with momentum
├── train.py         # Training script
├── predict.py       # Prediction script
├── analysis.py      # Performance analysis
├── models/          # Saved models (.pkl)
├── results/         # Saved metrics (.json)
├── plots/           # Saved training plots (.png)
├── predicted_images/# Saved prediction sample grids (.png)
├── requirements.txt # Dependencies
├── report.md        # Complete project report
└── README.md        # This file
```

## Features

### Autograd Engine (`autograd.py`)
- Reverse-mode automatic differentiation
- Operations: +, -, *, /, **, exp, log, sin, cos, tanh, ReLU, sigmoid, matmul, transpose, reshape, sum, mean, max
- NumPy-style broadcasting
- Computational graph tracking

### Neural Network (`nn.py`)
- Linear layers with Xavier initialization
- Activation functions: ReLU, Tanh, Sigmoid
- Cross-entropy loss with numerical stability
- Sequential container

### Optimizer (`optim.py`)
- SGD with momentum (0.9)
- Learning rate scheduling
- Gradient management

### Training (`train.py`)
- MNIST / Fashion-MNIST / CIFAR-10 (grayscale) dataset loading
- Training loop with validation
- Learning rate decay
- Saves **separate plots** to `plots/`
- Saves results JSON to `results/`
- Saves trained model to `models/`

## Model Architecture

- **Input:** 784 features (28×28 pixels)
- **Hidden Layers:** 3 × 128 neurons
- **Output:** 10 classes (digits 0-9)
- **Total Parameters:** 134,794
- **Activation:** ReLU or Tanh (compared)

## Results

After training, check:
- **Plots** (saved separately):
  - `plots/loss_curves_<activation>_<dataset>.png`
  - `plots/accuracy_curves_<activation>_<dataset>.png`
  - `plots/test_loss_<activation>_<dataset>.png`
  - `plots/test_accuracy_<activation>_<dataset>.png`
  - `plots/lr_schedule_<activation>_<dataset>.png`
  - `plots/overfitting_gap_<activation>_<dataset>.png`
- **Results JSON**:
  - `results/results_<activation>_<dataset>.json`
- **Models**:
  - `models/model_<activation>_<dataset>.pkl` (overwritten each train run)
- **Prediction visualizations**:
  - `predicted_images/sample_predictions_*.png`

## Report

See `report.md` for complete documentation including:
- Dataset details
- Mathematical model derivation
- Training details
- Analysis and results
- Complete code documentation

## License

This project is for educational purposes.

