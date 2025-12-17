import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from autograd import Tensor
from nn import Sequential, Linear, ReLU, Tanh, cross_entropy_loss
from optim import SGD
import time
import os
import json


def load_mnist():
    """
    Load MNIST handwritten digits dataset.
    
    Dataset Details:
    - Source: OpenML (via scikit-learn)
    - Total samples: 70,000
    - Features: 784 (28x28 grayscale images)
    - Classes: 10 (digits 0-9)
    - Feature range: [0, 255] -> normalized to [0, 1]
    """
    print("=" * 60)
    print("DATASET: MNIST Handwritten Digits")
    print("=" * 60)
    print("Loading dataset from OpenML...")
    
    mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
    X, y = mnist.data, mnist.target.astype(np.int32)
    
    print(f"Raw data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Feature range (before scaling): [{X.min():.1f}, {X.max():.1f}]")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Normalize to [0, 1] by dividing by 255
    X = X / 255.0
    
    print(f"Feature range (after scaling): [{X.min():.2f}, {X.max():.2f}]")
    print(f"Data type: {X.dtype}")
    
    # Split into train and test (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test


def load_fashion_mnist():
    """
    Load Fashion-MNIST dataset.
    
    Dataset Details:
    - Source: OpenML (via scikit-learn)
    - Total samples: 70,000
    - Features: 784 (28x28 grayscale images)
    - Classes: 10 (fashion categories)
    - Feature range: [0, 255] -> normalized to [0, 1]
    """
    print("=" * 60)
    print("DATASET: Fashion-MNIST")
    print("=" * 60)
    print("Loading dataset from OpenML...")
    
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1, parser='auto', as_frame=False)
    X, y = fashion_mnist.data, fashion_mnist.target.astype(np.int32)
    
    print(f"Raw data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Feature range (before scaling): [{X.min():.1f}, {X.max():.1f}]")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Normalize to [0, 1]
    X = X / 255.0
    
    print(f"Feature range (after scaling): [{X.min():.2f}, {X.max():.2f}]")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test


def load_cifar10_grayscale():
    """
    Load CIFAR-10 dataset (converted to grayscale).

    Dataset Details:
    - Source: OpenML (via scikit-learn)
    - Dataset: CIFAR_10_small
    - Total samples: 60,000
    - Features: 3072 (32x32x3) -> converted to 1024 (32x32 grayscale)
    - Classes: 10
    - Feature range: [0, 255] -> normalized to [0, 1]
    """
    print("=" * 60)
    print("DATASET: CIFAR-10 (Grayscale)")
    print("=" * 60)
    print("Loading dataset from OpenML...")

    cifar = fetch_openml('CIFAR_10_small', version=1, parser='auto', as_frame=False)
    X, y = cifar.data, cifar.target.astype(np.int32)

    print(f"Raw data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Feature range (before scaling): [{X.min():.1f}, {X.max():.1f}]")
    print(f"Number of classes: {len(np.unique(y))}")

    # Normalize to [0, 1]
    X = X.astype(np.float32) / 255.0

    # CIFAR-10_small is typically flattened as [R(1024), G(1024), B(1024)]
    r = X[:, :1024].reshape(-1, 32, 32)
    g = X[:, 1024:2048].reshape(-1, 32, 32)
    b = X[:, 2048:].reshape(-1, 32, 32)

    # Convert to grayscale
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    X_gray = gray.reshape(-1, 1024)

    print(f"Grayscale data shape: {X_gray.shape}")
    print(f"Feature range (after scaling): [{X_gray.min():.2f}, {X_gray.max():.2f}]")

    X_train, X_test, y_train, y_test = train_test_split(
        X_gray, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print("=" * 60)

    return X_train, X_test, y_train, y_test


def accuracy(logits, targets):
    """Compute classification accuracy"""
    predictions = np.argmax(logits.data, axis=1)
    return np.mean(predictions == targets)


def train_epoch(model, optimizer, X_train, y_train, batch_size=64):
    """
    Train for one epoch.
    
    Returns:
        avg_loss: Average training loss
        avg_acc: Average training accuracy
        n_iterations: Number of iterations (batches) in this epoch
    """
    n_samples = X_train.shape[0]
    indices = np.random.permutation(n_samples)
    
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = Tensor(X_train[batch_indices], requires_grad=False)
        y_batch = y_train[batch_indices]
        
        # Forward pass
        logits = model(X_batch)
        loss, probs = cross_entropy_loss(logits, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_acc += accuracy(logits, y_batch)
        n_batches += 1
    
    return total_loss / n_batches, total_acc / n_batches, n_batches


def evaluate(model, X_data, y_data, batch_size=64):
    """
    Evaluate model on a dataset.
    
    Returns:
        avg_loss: Average loss
        avg_acc: Average accuracy
    """
    n_samples = X_data.shape[0]
    
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    
    for i in range(0, n_samples, batch_size):
        X_batch = Tensor(X_data[i:i+batch_size], requires_grad=False)
        y_batch = y_data[i:i+batch_size]
        
        logits = model(X_batch)
        loss, probs = cross_entropy_loss(logits, y_batch)
        
        total_loss += loss.item()
        total_acc += accuracy(logits, y_batch)
        n_batches += 1
    
    return total_loss / n_batches, total_acc / n_batches


def train_model(activation='relu', dataset='mnist', epochs=30, lr=0.01, 
                batch_size=64, use_cv=False, cv_folds=5):
    """
    Train a neural network model.
    
    Args:
        activation: 'relu' or 'tanh'
        dataset: 'mnist' or 'fashion'
        epochs: Number of training epochs
        lr: Initial learning rate
        batch_size: Batch size for training
        use_cv: Whether to use cross-validation
        cv_folds: Number of CV folds if use_cv=True
    
    Returns:
        Dictionary containing training history and model
    """
    
    # Load data
    if dataset == 'mnist':
        X_train, X_test, y_train, y_test = load_mnist()
    elif dataset == 'fashion':
        X_train, X_test, y_train, y_test = load_fashion_mnist()
    elif dataset == 'cifar10':
        X_train, X_test, y_train, y_test = load_cifar10_grayscale()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    # Build model
    if activation.lower() == 'relu':
        act_fn = ReLU
    elif activation.lower() == 'tanh':
        act_fn = Tanh
    else:
        raise ValueError(f"Unknown activation: {activation}")
    
    model = Sequential(
        Linear(input_size, 128),
        act_fn(),
        Linear(128, 128),
        act_fn(),
        Linear(128, 128),
        act_fn(),
        Linear(128, num_classes)
    )
    
    # Count parameters
    n_params = sum(p.data.size for p in model.parameters)
    
    print(f"\nMODEL ARCHITECTURE:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden layers: 3 x 128 neurons")
    print(f"  Output size: {num_classes}")
    print(f"  Activation: {activation.upper()}")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Initial learning rate: {lr}")
    print(f"  Momentum: 0.9")
    
    # Optimizer with momentum 0.9
    optimizer = SGD(model.parameters, lr=lr, momentum=0.9)
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    test_losses = []
    test_accs = []
    lr_history = []
    iterations_per_epoch = []
    total_iterations = 0
    
    if use_cv:
        # Cross-validation setup
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        X_train_cv = X_train
        y_train_cv = y_train
        print(f"\nUsing {cv_folds}-fold cross-validation")
    else:
        # Simple train/validation split
        X_train_cv, X_val, y_train_cv, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )
        print(f"\nTrain/Validation split: {X_train_cv.shape[0]}/{X_val.shape[0]} samples")
    
    print(f"\n{'='*60}")
    print(f"TRAINING: {activation.upper()} activation on {dataset.upper()}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        # Learning rate schedule: decay by 0.1 at epoch 15 and 25
        if epoch == 15:
            optimizer.lr *= 0.1
            print(f"Learning rate reduced to {optimizer.lr:.6f}")
        if epoch == 25:
            optimizer.lr *= 0.1
            print(f"Learning rate reduced to {optimizer.lr:.6f}")

        lr_history.append(float(optimizer.lr))
        
        # Train
        train_loss, train_acc, n_iter = train_epoch(
            model, optimizer, X_train_cv, y_train_cv, batch_size=batch_size
        )
        total_iterations += n_iter
        iterations_per_epoch.append(n_iter)
        
        # Validation
        if use_cv:
            # For CV, evaluate on test set as validation proxy
            val_loss, val_acc = evaluate(model, X_test, y_test, batch_size=batch_size)
        else:
            val_loss, val_acc = evaluate(model, X_val, y_val, batch_size=batch_size)
        
        # Test evaluation
        test_loss, test_acc = evaluate(model, X_test, y_test, batch_size=batch_size)
        
        # Record history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    best_test_acc = max(test_accs)
    best_test_epoch = np.argmax(test_accs) + 1
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total iterations: {total_iterations:,}")
    print(f"Best test accuracy: {best_test_acc:.4f} (at epoch {best_test_epoch})")
    print(f"Final test accuracy: {test_accs[-1]:.4f}")
    print(f"{'='*60}\n")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'lr_history': lr_history,
        'best_test_acc': best_test_acc,
        'best_test_epoch': best_test_epoch,
        'total_iterations': total_iterations,
        'iterations_per_epoch': iterations_per_epoch,
        'n_params': n_params,
        'activation': activation,
        'dataset': dataset
    }


def plot_training_curves(results, save_dir='plots'):
    """Plot and save training curves as separate figures"""
    os.makedirs(save_dir, exist_ok=True)
    
    activation = results['activation']
    dataset = results['dataset']
    epochs = range(1, len(results['train_losses']) + 1)

    prefix = f"{activation}_{dataset}"

    # 1) Loss curves (train/val/test)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, results['train_losses'], label='Train Loss', linewidth=2)
    plt.plot(epochs, results['val_losses'], label='Validation Loss', linewidth=2)
    plt.plot(epochs, results['test_losses'], label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{activation.upper()} - Loss Curves ({dataset.upper()})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    loss_path = os.path.join(save_dir, f"loss_curves_{prefix}.png")
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {loss_path}")

    # 2) Accuracy curves (train/val/test)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, results['train_accs'], label='Train Accuracy', linewidth=2)
    plt.plot(epochs, results['val_accs'], label='Validation Accuracy', linewidth=2)
    plt.plot(epochs, results['test_accs'], label='Test Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title(f'{activation.upper()} - Accuracy Curves ({dataset.upper()})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    acc_path = os.path.join(save_dir, f"accuracy_curves_{prefix}.png")
    plt.savefig(acc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {acc_path}")

    # 3) Test loss only
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, results['test_losses'], label='Test Loss', linewidth=2, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title(f'{activation.upper()} - Test Loss ({dataset.upper()})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    test_loss_path = os.path.join(save_dir, f"test_loss_{prefix}.png")
    plt.savefig(test_loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {test_loss_path}")

    # 4) Test accuracy only
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, results['test_accs'], label='Test Accuracy', linewidth=2, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1)
    plt.title(f'{activation.upper()} - Test Accuracy ({dataset.upper()})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    test_acc_path = os.path.join(save_dir, f"test_accuracy_{prefix}.png")
    plt.savefig(test_acc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {test_acc_path}")

    # 5) Learning rate schedule (if available)
    if 'lr_history' in results and results['lr_history']:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, results['lr_history'], linewidth=2, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(f'Learning Rate Schedule ({activation.upper()} on {dataset.upper()})')
        plt.grid(True, alpha=0.3)
        lr_path = os.path.join(save_dir, f"lr_schedule_{prefix}.png")
        plt.savefig(lr_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {lr_path}")

    # 6) Overfitting gap: (test loss - train loss)
    if len(results['train_losses']) == len(results['test_losses']):
        gap = np.array(results['test_losses']) - np.array(results['train_losses'])
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, gap, linewidth=2, color='purple')
        plt.axhline(0.0, linestyle='--', linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('Test Loss - Train Loss')
        plt.title(f'Overfitting Gap ({activation.upper()} on {dataset.upper()})')
        plt.grid(True, alpha=0.3)
        gap_path = os.path.join(save_dir, f"overfitting_gap_{prefix}.png")
        plt.savefig(gap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {gap_path}")


def save_results(results, save_path='results.json'):
    """Save training results to JSON file"""
    def _to_jsonable(obj):
        # numpy scalars
        if isinstance(obj, np.generic):
            return obj.item()
        # numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # lists/tuples
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(x) for x in obj]
        # dicts
        if isinstance(obj, dict):
            return {str(k): _to_jsonable(v) for k, v in obj.items()}
        # plain python types (int/float/str/bool/None)
        return obj

    # Remove model (can't pickle to JSON)
    results_copy = {k: v for k, v in results.items() if k != 'model'}
    results_copy['model_info'] = {
        'n_params': results.get('n_params'),
        'activation': results.get('activation'),
        'dataset': results.get('dataset'),
    }

    results_copy = _to_jsonable(results_copy)

    with open(save_path, 'w') as f:
        json.dump(results_copy, f, indent=2)
    print(f"Results saved to {save_path}")


if __name__ == '__main__':
    # Optimal parameters (determined through experimentation)
    OPTIMAL_PARAMS = {
        'epochs': 30,
        'lr': 0.01,
        'batch_size': 64,
        'momentum': 0.9,
        'lr_decay_epochs': [15, 25],
        'lr_decay_factor': 0.1
    }
    
    print("OPTIMAL HYPERPARAMETERS:")
    print(json.dumps(OPTIMAL_PARAMS, indent=2))
    print()
    
    # Train with ReLU
    print("\n" + "="*60)
    results_relu = train_model(
        activation='relu',
        dataset='mnist',
        epochs=OPTIMAL_PARAMS['epochs'],
        lr=OPTIMAL_PARAMS['lr'],
        batch_size=OPTIMAL_PARAMS['batch_size'],
        use_cv=False
    )
    plot_training_curves(results_relu)
    save_results(results_relu, 'results_relu.json')
    
    # Train with Tanh
    print("\n" + "="*60)
    results_tanh = train_model(
        activation='tanh',
        dataset='mnist',
        epochs=OPTIMAL_PARAMS['epochs'],
        lr=OPTIMAL_PARAMS['lr'],
        batch_size=OPTIMAL_PARAMS['batch_size'],
        use_cv=False
    )
    plot_training_curves(results_tanh)
    save_results(results_tanh, 'results_tanh.json')



