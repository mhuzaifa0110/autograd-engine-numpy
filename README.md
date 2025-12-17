# Automatic Differentiation Engine from Scratch

A complete implementation of automatic differentiation (autograd) engine using only NumPy, and training neural networks with it.

## Project Overview

This project implements:
1. **Automatic Differentiation Engine** - Reverse-mode AD with full NumPy compatibility
2. **Neural Network Training** - MLP with 3 hidden layers (128 neurons each)
3. **Real Dataset Training** - MNIST handwritten digits classification
4. **Complete Analysis** - Performance metrics, Hessian computation, activation comparison

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
# Train a ReLU model
python main.py train --activation relu --epochs 30

# Train a Tanh model
python main.py train --activation tanh --epochs 30 --save-model

# Compare both activations
python main.py compare --epochs 30
```

**Option B: Using train.py directly**
```bash
python train.py
```

This will:
- Train two models (ReLU and Tanh activations)
- Generate training curves
- Save results to JSON files

### 3. Make Predictions

**Using main.py:**
```bash
python main.py predict --model model_relu_mnist.pkl --show-examples 20
```

**Or using Python directly:**
```python
from predict import load_model, predict
from train import load_mnist

model = load_model('model.pkl')
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
python main.py analyze --type hessian --model model_relu_mnist.pkl
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
- MNIST dataset loading
- Training loop with validation
- Learning rate decay
- Comprehensive visualization
- Results saving

## Model Architecture

- **Input:** 784 features (28×28 pixels)
- **Hidden Layers:** 3 × 128 neurons
- **Output:** 10 classes (digits 0-9)
- **Total Parameters:** 134,794
- **Activation:** ReLU or Tanh (compared)

## Results

After training, check:
- `plots/training_curves_relu_mnist.png` - ReLU training curves
- `plots/training_curves_tanh_mnist.png` - Tanh training curves
- `results_relu.json` - ReLU training history
- `results_tanh.json` - Tanh training history

## Report

See `report.md` for complete documentation including:
- Dataset details
- Mathematical model derivation
- Training details
- Analysis and results
- Complete code documentation

## License

This project is for educational purposes.

