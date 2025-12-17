"""
Analysis Script - Performance Analysis and Hessian Computation
==============================================================

This script provides analysis tools for:
1. Measuring backward() performance (time and memory)
2. Scaling behavior analysis
3. Hessian eigenvalue computation
4. Activation function comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from autograd import Tensor
from nn import Sequential, Linear, ReLU, Tanh, cross_entropy_loss
import time
import tracemalloc


def measure_backward_performance(model, input_size=784, batch_size=64):
    """
    Measure time and memory usage of backward pass.
    
    Args:
        model: Neural network model
        input_size: Input feature size
        batch_size: Batch size for testing
    
    Returns:
        Dictionary with performance metrics
    """
    print("=" * 60)
    print("MEASURING BACKWARD() PERFORMANCE")
    print("=" * 60)
    
    # Create dummy input
    X = Tensor(np.random.randn(batch_size, input_size).astype(np.float32), requires_grad=False)
    
    # Forward pass
    print("Running forward pass...")
    start_time = time.time()
    tracemalloc.start()
    
    logits = model(X)
    loss = logits.sum()
    
    forward_time = time.time() - start_time
    forward_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    
    # Backward pass
    print("Running backward pass...")
    tracemalloc.start()
    start_time = time.time()
    
    loss.backward()
    
    backward_time = time.time() - start_time
    backward_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    
    print(f"\nResults:")
    print(f"  Forward time: {forward_time:.4f}s")
    print(f"  Backward time: {backward_time:.4f}s")
    print(f"  Forward memory: {forward_memory / 1024 / 1024:.2f} MB")
    print(f"  Backward memory: {backward_memory / 1024 / 1024:.2f} MB")
    print(f"  Speedup ratio (forward/backward): {forward_time/backward_time:.2f}x")
    print("=" * 60)
    
    return {
        'forward_time': forward_time,
        'backward_time': backward_time,
        'forward_memory': forward_memory,
        'backward_memory': backward_memory
    }


def compute_hessian_eigenvalues(model, X_sample, y_sample, max_params=500):
    """
    Compute Hessian eigenvalues (approximate using finite differences).
    This is computationally expensive, so we limit to a subset of parameters.
    
    Args:
        model: Trained neural network model
        X_sample: Sample input data
        y_sample: Sample labels
        max_params: Maximum number of parameters to include in Hessian
    
    Returns:
        eigenvals: Sorted eigenvalues (descending)
    """
    print("=" * 60)
    print("COMPUTING HESSIAN EIGENVALUES")
    print("=" * 60)
    
    # Flatten all parameters
    params = model.parameters
    n_params = sum(p.data.size for p in params)
    
    print(f"Total parameters: {n_params:,}")
    print(f"Computing Hessian for first {max_params} parameters...")
    
    # Convert sample to tensor
    X_tensor = Tensor(X_sample, requires_grad=False)
    
    # Compute initial gradient
    logits = model(X_tensor)
    loss, _ = cross_entropy_loss(logits, y_sample)
    
    # Zero gradients
    for p in params:
        p.zero_grad()
    
    loss.backward()
    
    # Collect gradients
    grads = []
    for p in params:
        grads.append(p.grad.flatten())
    grad_flat = np.concatenate(grads)
    
    # Limit to max_params
    n_compute = min(max_params, n_params)
    grad_flat = grad_flat[:n_compute]
    
    # Approximate Hessian using finite differences
    epsilon = 1e-5
    hessian = np.zeros((n_compute, n_compute))
    
    print(f"Computing Hessian matrix ({n_compute}x{n_compute})...")
    
    # Flatten all parameters for easy indexing
    param_flat_list = []
    param_shapes = []
    param_offsets = []
    offset = 0
    
    for p in params:
        flat = p.data.flatten()
        param_flat_list.append(flat)
        param_shapes.append(p.data.shape)
        param_offsets.append(offset)
        offset += flat.size
    
    all_params_flat = np.concatenate(param_flat_list)
    
    for i in range(n_compute):
        if i % 50 == 0:
            print(f"  Progress: {i}/{n_compute}")
        
        # Perturb parameter i
        original_val = all_params_flat[i]
        all_params_flat[i] += epsilon
        
        # Restore to parameter tensors
        offset_idx = 0
        for p_idx, p in enumerate(params):
            if i < param_offsets[p_idx] + p.data.size:
                local_idx = i - param_offsets[p_idx]
                flat_idx = np.unravel_index(local_idx, p.data.shape)
                p.data[flat_idx] = all_params_flat[i]
                break
        
        # Recompute gradient
        for p in params:
            p.zero_grad()
        logits = model(X_tensor)
        loss, _ = cross_entropy_loss(logits, y_sample)
        loss.backward()
        
        # Collect new gradients
        new_grads = []
        for p in params:
            new_grads.append(p.grad.flatten())
        new_grad_flat = np.concatenate(new_grads)[:n_compute]
        
        # Compute Hessian column
        hessian[:, i] = (new_grad_flat - grad_flat) / epsilon
        
        # Restore parameter
        all_params_flat[i] = original_val
        offset_idx = 0
        for p_idx, p in enumerate(params):
            if i < param_offsets[p_idx] + p.data.size:
                local_idx = i - param_offsets[p_idx]
                flat_idx = np.unravel_index(local_idx, p.data.shape)
                p.data[flat_idx] = original_val
                break
    
    # Compute eigenvalues
    print("Computing eigenvalues...")
    eigenvals = np.linalg.eigvalsh(hessian)
    eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
    
    print(f"Done! Computed {len(eigenvals)} eigenvalues")
    print(f"  Largest eigenvalue: {eigenvals[0]:.2e}")
    print(f"  Smallest eigenvalue: {eigenvals[-1]:.2e}")
    print(f"  Condition number: {eigenvals[0] / (eigenvals[-1] + 1e-10):.2e}")
    print("=" * 60)
    
    return eigenvals


def plot_hessian_eigenvalues(eigenvals_relu, eigenvals_tanh, save_path='hessian_eigenvalues.png'):
    """Plot Hessian eigenvalue distributions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Eigenvalue spectrum
    ax1.semilogy(eigenvals_relu, label='ReLU', linewidth=2, color='blue')
    ax1.semilogy(eigenvals_tanh, label='Tanh', linewidth=2, color='orange')
    ax1.set_xlabel('Eigenvalue Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue (log scale)', fontsize=12)
    ax1.set_title('Hessian Eigenvalue Spectrum', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Condition number
    cond_relu = eigenvals_relu[0] / (eigenvals_relu[-1] + 1e-10)
    cond_tanh = eigenvals_tanh[0] / (eigenvals_tanh[-1] + 1e-10)
    
    ax2.bar(['ReLU', 'Tanh'], [cond_relu, cond_tanh], color=['blue', 'orange'], alpha=0.7)
    ax2.set_ylabel('Condition Number', fontsize=12)
    ax2.set_title('Hessian Condition Number', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved Hessian plot to {save_path}")
    plt.close()


def scaling_analysis():
    """Analyze scaling behavior of backward pass with different batch sizes"""
    print("=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)
    
    batch_sizes = [16, 32, 64, 128, 256]
    times = []
    memories = []
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        model = Sequential(
            Linear(784, 128),
            ReLU(),
            Linear(128, 128),
            ReLU(),
            Linear(128, 10)
        )
        
        X = Tensor(np.random.randn(batch_size, 784).astype(np.float32), requires_grad=False)
        
        logits = model(X)
        loss = logits.sum()
        
        start_time = time.time()
        tracemalloc.start()
        loss.backward()
        backward_time = time.time() - start_time
        memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        
        times.append(backward_time)
        memories.append(memory / 1024 / 1024)  # MB
        
        print(f"  Time: {backward_time:.4f}s, Memory: {memory/1024/1024:.2f} MB")
    
    # Plot scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(batch_sizes, times, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Backward Time (s)', fontsize=12)
    ax1.set_title('Time Scaling', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(batch_sizes, memories, 'o-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax2.set_title('Memory Scaling', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scaling_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved scaling analysis plot to scaling_analysis.png")
    plt.close()
    
    # Analyze scaling behavior
    print("\nScaling Analysis:")
    print(f"  Time scaling factor: {times[-1]/times[0]:.2f}x for {batch_sizes[-1]/batch_sizes[0]:.1f}x batch size")
    print(f"  Memory scaling factor: {memories[-1]/memories[0]:.2f}x for {batch_sizes[-1]/batch_sizes[0]:.1f}x batch size")
    print("=" * 60)


if __name__ == '__main__':
    # Example: Measure performance
    print("\n" + "="*60)
    print("PERFORMANCE MEASUREMENT")
    print("="*60)
    
    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 128),
        ReLU(),
        Linear(128, 10)
    )
    
    measure_backward_performance(model, input_size=784, batch_size=64)
    
    # Scaling analysis
    print("\n")
    scaling_analysis()
    
    print("\n" + "="*60)
    print("NOTE: Hessian computation is expensive.")
    print("To compute Hessian eigenvalues, use:")
    print("  eigenvals = compute_hessian_eigenvalues(model, X_sample, y_sample)")
    print("="*60)



