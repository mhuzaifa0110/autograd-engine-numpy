"""
Prediction Code - Annex C
=========================

This script provides functions for making predictions using trained models.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from autograd import Tensor
from nn import Sequential, Linear, ReLU, Tanh


def load_model(model_path='model.pkl'):
    """
    Load a saved model from disk.
    
    Args:
        model_path: Path to the saved model file (.pkl)
    
    Returns:
        model: Trained neural network model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict(model, X, batch_size=64):
    """
    Make predictions on new data.
    
    Args:
        model: Trained neural network model
        X: Input data (numpy array), shape (n_samples, n_features)
        batch_size: Batch size for prediction
    
    Returns:
        predictions: Predicted class labels (numpy array), shape (n_samples,)
        probabilities: Class probabilities (numpy array), shape (n_samples, n_classes)
    """
    n_samples = X.shape[0]
    all_predictions = []
    all_probs = []
    
    for i in range(0, n_samples, batch_size):
        X_batch = Tensor(X[i:i+batch_size], requires_grad=False)
        
        # Forward pass
        logits = model(X_batch)
        
        # Compute probabilities (softmax)
        max_logits = logits.max(axis=1, keepdims=True)
        logits_stable = logits - max_logits
        exp_logits = logits_stable.exp()
        sum_exp = exp_logits.sum(axis=1, keepdims=True)
        probs = exp_logits / sum_exp
        
        # Get predictions
        predictions = np.argmax(logits.data, axis=1)
        
        all_predictions.append(predictions)
        all_probs.append(probs.data)
    
    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probs)
    
    return predictions, probabilities


def predict_single(model, x):
    """
    Predict for a single sample.
    
    Args:
        model: Trained model
        x: Single sample (numpy array), shape (n_features,)
    
    Returns:
        predicted_class: Predicted class (int)
        probabilities: Class probabilities (numpy array), shape (n_classes,)
    """
    x = x.reshape(1, -1)  # Add batch dimension
    predictions, probabilities = predict(model, x, batch_size=1)
    return predictions[0], probabilities[0]


def evaluate_predictions(y_true, y_pred):
    """
    Evaluate prediction accuracy.
    
    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
    
    Returns:
        accuracy: Classification accuracy (float)
    """
    return np.mean(y_true == y_pred)


def save_model(model, model_path='model.pkl'):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained neural network model
        model_path: Path to save the model (.pkl)
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")


def visualize_samples(X, y_true, y_pred, probabilities, n_samples=10, save_path=None):
    """
    Visualize sample predictions with images.
    
    Args:
        X: Input images (n_samples, 784) - normalized to [0, 1]
        y_true: True labels
        y_pred: Predicted labels
        probabilities: Prediction probabilities
        n_samples: Number of samples to display
        save_path: Optional path to save the figure
    """
    n_display = min(n_samples, 10)  # Display up to 10 samples
    rows = 2
    cols = 5
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(n_display):
        # Reshape image from 784 to 28x28
        img = X[i].reshape(28, 28)
        
        # Display image
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        
        # Set title with prediction info
        pred = y_pred[i]
        true = y_true[i]
        conf = probabilities[i][pred]
        color = 'green' if pred == true else 'red'
        axes[i].set_title(f'True: {true}\nPred: {pred} ({conf:.2f})', 
                         color=color, fontsize=10, fontweight='bold')
    
    # Hide unused subplots
    for i in range(n_display, rows * cols):
        axes[i].axis('off')
    
    # Add main title with proper spacing
    fig.suptitle('Sample Predictions - MNIST Digits', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for title
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


if __name__ == '__main__':
    # Example usage
    
    print("=" * 60)
    print("PREDICTION CODE - EXAMPLE USAGE")
    print("=" * 60)
    
    # Example: Load test data
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    
    print("\nLoading test data...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
    X, y = mnist.data, mnist.target.astype(np.int32)
    X = X / 255.0
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of test samples: {len(y_test)}")
    
    print("\n" + "=" * 60)
    print("NOTE: To use this script with a trained model:")
    print("=" * 60)
    print("1. Train your model using train.py")
    print("2. Save the model during training:")
    print("   from predict import save_model")
    print("   save_model(model, 'model_relu.pkl')")
    print("3. Load and use:")
    print("   model = load_model('model_relu.pkl')")
    print("   predictions, probs = predict(model, X_test)")
    print("   accuracy = evaluate_predictions(y_test, predictions)")
    print("=" * 60)



