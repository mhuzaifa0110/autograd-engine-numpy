import numpy as np
from typing import Optional, Callable, Tuple, List


class Tensor:
    """
    A Tensor class with automatic differentiation using reverse-mode AD.
    Implements a computational graph that tracks operations and computes gradients.
    """
    
    def __init__(self, data, requires_grad=False, _op=None, _children=(), _backward_fn=None):
        """
        Args:
            data: numpy array or scalar
            requires_grad: whether to track gradients for this tensor
            _op: operation that created this tensor (for graph visualization)
            _children: child tensors in the computational graph
            _backward_fn: function to compute gradients during backward pass
        """
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._op = _op
        self._backward_fn = _backward_fn
        self._children = set(_children)
        
        if requires_grad:
            self.grad = np.zeros_like(self.data)
    
    def backward(self, grad=None):
        """
        Backward pass through the computational graph.
        Computes gradients using reverse-mode automatic differentiation.
        """
        if not self.requires_grad:
            return
        
        if grad is None:
            grad = np.ones_like(self.data)
        
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        
        self.grad += grad
        
        # Topological sort for reverse order traversal
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Backward pass in reverse topological order
        self.grad = grad
        for node in reversed(topo):
            if node._backward_fn is not None:
                node._backward_fn()
    
    def zero_grad(self):
        """Reset gradients to zero."""
        if self.grad is not None:
            self.grad.fill(0)
    
    def detach(self):
        """Return a new tensor detached from the computational graph."""
        return Tensor(self.data.copy(), requires_grad=False)
    
    def item(self):
        """Return scalar value if tensor is a scalar."""
        return self.data.item()
    
    def numpy(self):
        """Return numpy array."""
        return self.data
    
    @property
    def shape(self):
        """Return tensor shape."""
        return self.data.shape
    
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    # ========== Unary Operations ==========
    
    def __neg__(self):
        return neg(self)
    
    def exp(self):
        return exp(self)
    
    def log(self):
        return log(self)
    
    def sin(self):
        return sin(self)
    
    def cos(self):
        return cos(self)
    
    def tanh(self):
        return tanh(self)
    
    def relu(self):
        return relu(self)
    
    def sigmoid(self):
        return sigmoid(self)
    
    def sum(self, axis=None, keepdims=False):
        return sum_op(self, axis=axis, keepdims=keepdims)
    
    def mean(self, axis=None, keepdims=False):
        return mean_op(self, axis=axis, keepdims=keepdims)
    
    def max(self, axis=None, keepdims=False):
        return max_op(self, axis=axis, keepdims=keepdims)
    
    def transpose(self, axes=None):
        return transpose(self, axes=axes)
    
    def reshape(self, shape):
        return reshape(self, shape)
    
    # ========== Binary Operations ==========
    
    def __add__(self, other):
        return add(self, other)
    
    def __radd__(self, other):
        return add(other, self)
    
    def __sub__(self, other):
        return sub(self, other)
    
    def __rsub__(self, other):
        return sub(other, self)
    
    def __mul__(self, other):
        return mul(self, other)
    
    def __rmul__(self, other):
        return mul(other, self)
    
    def __truediv__(self, other):
        return div(self, other)
    
    def __rtruediv__(self, other):
        return div(other, self)
    
    def __pow__(self, other):
        return pow_op(self, other)
    
    def __matmul__(self, other):
        return matmul(self, other)
    
    def __rmatmul__(self, other):
        return matmul(other, self)


# ========== Operation Functions ==========

def _ensure_tensor(x):
    """Convert numpy array or scalar to Tensor if needed."""
    if not isinstance(x, Tensor):
        return Tensor(x, requires_grad=False)
    return x


def add(a, b):
    """Addition: a + b"""
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    out = Tensor(
        a.data + b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _op='+',
        _children=(a, b)
    )
    
    def _backward():
        if a.requires_grad:
            # Handle broadcasting: sum over dimensions that were broadcasted
            grad_a = out.grad
            # Sum over extra dimensions
            while grad_a.ndim > a.data.ndim:
                grad_a = grad_a.sum(axis=0)
            # Sum over dimensions where a had size 1 but grad has larger size
            for i in range(min(grad_a.ndim, a.data.ndim)):
                if a.data.shape[i] == 1 and grad_a.shape[i] > 1:
                    grad_a = grad_a.sum(axis=i, keepdims=True)
            # Ensure shape matches
            grad_a = np.broadcast_to(grad_a, a.shape).reshape(a.shape)
            a.grad += grad_a
        if b.requires_grad:
            # Handle broadcasting: sum over dimensions that were broadcasted
            grad_b = out.grad
            # Sum over extra dimensions
            while grad_b.ndim > b.data.ndim:
                grad_b = grad_b.sum(axis=0)
            # Sum over dimensions where b had size 1 but grad has larger size
            for i in range(min(grad_b.ndim, b.data.ndim)):
                if b.data.shape[i] == 1 and grad_b.shape[i] > 1:
                    grad_b = grad_b.sum(axis=i, keepdims=True)
            # Ensure shape matches
            grad_b = np.broadcast_to(grad_b, b.shape).reshape(b.shape)
            b.grad += grad_b
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def sub(a, b):
    """Subtraction: a - b"""
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    out = Tensor(
        a.data - b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _op='-',
        _children=(a, b)
    )
    
    def _backward():
        if a.requires_grad:
            # Handle broadcasting: sum over dimensions that were broadcasted
            grad_a = out.grad
            # Sum over extra dimensions
            while grad_a.ndim > a.data.ndim:
                grad_a = grad_a.sum(axis=0)
            # Sum over dimensions where a had size 1 but grad has larger size
            for i in range(min(grad_a.ndim, a.data.ndim)):
                if a.data.shape[i] == 1 and grad_a.shape[i] > 1:
                    grad_a = grad_a.sum(axis=i, keepdims=True)
            # Ensure shape matches
            grad_a = np.broadcast_to(grad_a, a.shape).reshape(a.shape)
            a.grad += grad_a
        if b.requires_grad:
            # Handle broadcasting: sum over dimensions that were broadcasted
            grad_b = out.grad
            # Sum over extra dimensions
            while grad_b.ndim > b.data.ndim:
                grad_b = grad_b.sum(axis=0)
            # Sum over dimensions where b had size 1 but grad has larger size
            for i in range(min(grad_b.ndim, b.data.ndim)):
                if b.data.shape[i] == 1 and grad_b.shape[i] > 1:
                    grad_b = grad_b.sum(axis=i, keepdims=True)
            # Ensure shape matches
            grad_b = np.broadcast_to(grad_b, b.shape).reshape(b.shape)
            b.grad += -grad_b
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def mul(a, b):
    """Multiplication: a * b (element-wise)"""
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    out = Tensor(
        a.data * b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _op='*',
        _children=(a, b)
    )
    
    def _backward():
        if a.requires_grad:
            # Handle broadcasting
            grad_a = out.grad * b.data
            while grad_a.ndim > a.data.ndim:
                grad_a = grad_a.sum(axis=0)
            for i in range(min(grad_a.ndim, a.data.ndim)):
                if a.data.shape[i] == 1 and grad_a.shape[i] > 1:
                    grad_a = grad_a.sum(axis=i, keepdims=True)
            grad_a = np.broadcast_to(grad_a, a.shape).reshape(a.shape)
            a.grad += grad_a
        if b.requires_grad:
            # Handle broadcasting
            grad_b = out.grad * a.data
            while grad_b.ndim > b.data.ndim:
                grad_b = grad_b.sum(axis=0)
            for i in range(min(grad_b.ndim, b.data.ndim)):
                if b.data.shape[i] == 1 and grad_b.shape[i] > 1:
                    grad_b = grad_b.sum(axis=i, keepdims=True)
            grad_b = np.broadcast_to(grad_b, b.shape).reshape(b.shape)
            b.grad += grad_b
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def div(a, b):
    """Division: a / b (element-wise)"""
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    out = Tensor(
        a.data / b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _op='/',
        _children=(a, b)
    )
    
    def _backward():
        if a.requires_grad:
            # Handle broadcasting
            grad_a = out.grad / b.data
            while grad_a.ndim > a.data.ndim:
                grad_a = grad_a.sum(axis=0)
            for i in range(min(grad_a.ndim, a.data.ndim)):
                if a.data.shape[i] == 1 and grad_a.shape[i] > 1:
                    grad_a = grad_a.sum(axis=i, keepdims=True)
            grad_a = np.broadcast_to(grad_a, a.shape).reshape(a.shape)
            a.grad += grad_a
        if b.requires_grad:
            # Handle broadcasting
            grad_b = -out.grad * a.data / (b.data ** 2)
            while grad_b.ndim > b.data.ndim:
                grad_b = grad_b.sum(axis=0)
            for i in range(min(grad_b.ndim, b.data.ndim)):
                if b.data.shape[i] == 1 and grad_b.shape[i] > 1:
                    grad_b = grad_b.sum(axis=i, keepdims=True)
            grad_b = np.broadcast_to(grad_b, b.shape).reshape(b.shape)
            b.grad += grad_b
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def pow_op(a, b):
    """Power: a ** b"""
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    b_val = b.data if isinstance(b, Tensor) else b
    out = Tensor(
        a.data ** b_val,
        requires_grad=a.requires_grad,
        _op='**',
        _children=(a,)
    )
    
    def _backward():
        if a.requires_grad:
            grad_a = np.broadcast_to(out.grad * b_val * (a.data ** (b_val - 1)), a.shape).reshape(a.shape)
            a.grad += grad_a
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def matmul(a, b):
    """Matrix multiplication: a @ b"""
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    out = Tensor(
        a.data @ b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _op='@',
        _children=(a, b)
    )
    
    def _backward():
        if a.requires_grad:
            if out.grad.ndim == 1:
                grad_a = np.outer(out.grad, b.data)
            else:
                # For out = a @ b, grad_a = out.grad @ b.T
                grad_a = out.grad @ b.data.T
            # Ensure grad_a matches a's shape
            if grad_a.shape != a.shape:
                # If shapes don't match, try to fix by summing/reshaping
                if grad_a.size == a.data.size:
                    grad_a = grad_a.reshape(a.shape)
                else:
                    # Sum over extra dimensions
                    while grad_a.ndim > a.data.ndim:
                        grad_a = grad_a.sum(axis=0)
                    # Reshape if sizes match
                    if grad_a.size == a.data.size:
                        grad_a = grad_a.reshape(a.shape)
                    else:
                        # Last resort: broadcast and sum
                        grad_a = np.sum(grad_a, axis=tuple(range(grad_a.ndim - a.data.ndim)), keepdims=False)
                        grad_a = np.broadcast_to(grad_a, a.shape).reshape(a.shape)
            # Ensure a.grad is initialized and has correct shape
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            elif a.grad.shape != a.shape:
                a.grad = np.zeros_like(a.data)
            a.grad += grad_a
        if b.requires_grad:
            if out.grad.ndim == 1:
                grad_b = np.outer(a.data, out.grad)
            else:
                # For out = a @ b, grad_b = a.T @ out.grad
                grad_b = a.data.T @ out.grad
            # Ensure grad_b matches b's shape
            if grad_b.shape != b.shape:
                # If shapes don't match, try to fix by summing/reshaping
                if grad_b.size == b.data.size:
                    grad_b = grad_b.reshape(b.shape)
                else:
                    # Sum over extra dimensions
                    while grad_b.ndim > b.data.ndim:
                        grad_b = grad_b.sum(axis=0)
                    # Reshape if sizes match
                    if grad_b.size == b.data.size:
                        grad_b = grad_b.reshape(b.shape)
                    else:
                        # Last resort: broadcast and sum
                        grad_b = np.sum(grad_b, axis=tuple(range(grad_b.ndim - b.data.ndim)), keepdims=False)
                        grad_b = np.broadcast_to(grad_b, b.shape).reshape(b.shape)
            # Ensure b.grad is initialized and has correct shape
            if b.grad is None:
                b.grad = np.zeros_like(b.data)
            elif b.grad.shape != b.shape:
                b.grad = np.zeros_like(b.data)
            b.grad += grad_b
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def neg(a):
    """Negation: -a"""
    a = _ensure_tensor(a)
    out = Tensor(
        -a.data,
        requires_grad=a.requires_grad,
        _op='neg',
        _children=(a,)
    )
    
    def _backward():
        if a.requires_grad:
            a.grad += -out.grad
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def exp(a):
    """Exponential: exp(a)"""
    a = _ensure_tensor(a)
    out = Tensor(
        np.exp(a.data),
        requires_grad=a.requires_grad,
        _op='exp',
        _children=(a,)
    )
    
    def _backward():
        if a.requires_grad:
            a.grad += out.grad * out.data
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def log(a):
    """Natural logarithm: log(a)"""
    a = _ensure_tensor(a)
    out = Tensor(
        np.log(a.data + 1e-8),  # Small epsilon for numerical stability
        requires_grad=a.requires_grad,
        _op='log',
        _children=(a,)
    )
    
    def _backward():
        if a.requires_grad:
            a.grad += out.grad / (a.data + 1e-8)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def sin(a):
    """Sine: sin(a)"""
    a = _ensure_tensor(a)
    out = Tensor(
        np.sin(a.data),
        requires_grad=a.requires_grad,
        _op='sin',
        _children=(a,)
    )
    
    def _backward():
        if a.requires_grad:
            a.grad += out.grad * np.cos(a.data)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def cos(a):
    """Cosine: cos(a)"""
    a = _ensure_tensor(a)
    out = Tensor(
        np.cos(a.data),
        requires_grad=a.requires_grad,
        _op='cos',
        _children=(a,)
    )
    
    def _backward():
        if a.requires_grad:
            a.grad += out.grad * (-np.sin(a.data))
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def tanh(a):
    """Hyperbolic tangent: tanh(a)"""
    a = _ensure_tensor(a)
    out = Tensor(
        np.tanh(a.data),
        requires_grad=a.requires_grad,
        _op='tanh',
        _children=(a,)
    )
    
    def _backward():
        if a.requires_grad:
            a.grad += out.grad * (1 - out.data ** 2)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def relu(a):
    """ReLU activation: max(0, a)"""
    a = _ensure_tensor(a)
    out = Tensor(
        np.maximum(0, a.data),
        requires_grad=a.requires_grad,
        _op='relu',
        _children=(a,)
    )
    
    def _backward():
        if a.requires_grad:
            a.grad += out.grad * (a.data > 0).astype(np.float32)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def sigmoid(a):
    """Sigmoid activation: 1 / (1 + exp(-a))"""
    a = _ensure_tensor(a)
    out = Tensor(
        1 / (1 + np.exp(-a.data)),
        requires_grad=a.requires_grad,
        _op='sigmoid',
        _children=(a,)
    )
    
    def _backward():
        if a.requires_grad:
            a.grad += out.grad * out.data * (1 - out.data)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def sum_op(a, axis=None, keepdims=False):
    """Sum reduction"""
    a = _ensure_tensor(a)
    out = Tensor(
        np.sum(a.data, axis=axis, keepdims=keepdims),
        requires_grad=a.requires_grad,
        _op='sum',
        _children=(a,)
    )
    
    def _backward():
        if a.requires_grad:
            grad = np.array(out.grad)  # Ensure it's a numpy array
            if axis is not None and not keepdims:
                # Need to expand dimension at axis
                grad = np.expand_dims(grad, axis)
            elif not keepdims:
                # Scalar output, broadcast to all dimensions
                grad = np.ones_like(a.data) * grad
            else:
                # keepdims=True: grad has same ndim but reduced size at axis
                # Need to broadcast to original shape
                grad = np.broadcast_to(grad, a.shape)
            a.grad += grad
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def mean_op(a, axis=None, keepdims=False):
    """Mean reduction"""
    a = _ensure_tensor(a)
    out = Tensor(
        np.mean(a.data, axis=axis, keepdims=keepdims),
        requires_grad=a.requires_grad,
        _op='mean',
        _children=(a,)
    )
    
    def _backward():
        if a.requires_grad:
            grad = np.array(out.grad)  # Ensure it's a numpy array
            if axis is not None:
                n = a.data.shape[axis]
            else:
                n = a.data.size
            if axis is not None and not keepdims:
                # Need to expand dimension at axis
                grad = np.expand_dims(grad, axis)
            elif not keepdims:
                # Scalar output, broadcast to all dimensions
                grad = np.ones_like(a.data) * grad
            else:
                # keepdims=True: grad has same ndim but reduced size at axis
                # Need to broadcast to original shape
                grad = np.broadcast_to(grad, a.shape)
            a.grad += grad / n
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def max_op(a, axis=None, keepdims=False):
    """Max reduction"""
    a = _ensure_tensor(a)
    out = Tensor(
        np.max(a.data, axis=axis, keepdims=keepdims),
        requires_grad=a.requires_grad,
        _op='max',
        _children=(a,)
    )
    
    def _backward():
        if a.requires_grad:
            grad = np.array(out.grad)  # Ensure grad is a numpy array
            
            # Handle gradient shape based on axis and keepdims
            if axis is None:
                # Scalar output
                if not keepdims:
                    grad = np.ones_like(a.data) * grad
                else:
                    grad = np.broadcast_to(grad, a.shape)
                # Max mask for scalar case
                out_val = np.array(out.data).item() if np.isscalar(out.data) or (isinstance(out.data, np.ndarray) and out.data.ndim == 0) else out.data
                max_mask = (a.data == out_val)
            else:
                # Axis reduction
                if not keepdims:
                    # Need to expand dimension at axis
                    grad = np.expand_dims(grad, axis)
                else:
                    # keepdims=True: grad already has correct ndim, just broadcast
                    # grad shape should be same as out.data shape (e.g., (64, 1) for axis=1)
                    grad = np.broadcast_to(grad, a.shape)
                # Max mask for axis reduction
                if keepdims:
                    max_val = out.data  # Already has keepdims shape
                else:
                    max_val = np.expand_dims(out.data, axis)
                max_mask = (a.data == max_val)
            
            # Ensure all shapes match before operations
            # Flatten any extra dimensions that might have been introduced
            while grad.ndim > a.data.ndim:
                grad = grad.sum(axis=0)
            while max_mask.ndim > a.data.ndim:
                max_mask = max_mask.sum(axis=0)
            
            # Reshape to match a.shape
            if grad.shape != a.shape:
                if grad.size == a.data.size:
                    grad = grad.reshape(a.shape)
                else:
                    grad = np.broadcast_to(grad, a.shape)
            if max_mask.shape != a.shape:
                if max_mask.size == a.data.size:
                    max_mask = max_mask.reshape(a.shape)
                else:
                    max_mask = np.broadcast_to(max_mask, a.shape)
            
            # Ensure a.grad is initialized and has correct shape
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            elif a.grad.shape != a.shape:
                a.grad = np.zeros_like(a.data)
            
            # Compute final gradient contribution
            grad_contribution = grad * max_mask.astype(np.float32)
            # Ensure it matches a.grad shape
            if grad_contribution.shape != a.grad.shape:
                grad_contribution = grad_contribution.reshape(a.grad.shape)
            a.grad += grad_contribution
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def transpose(a, axes=None):
    """Transpose tensor"""
    a = _ensure_tensor(a)
    if axes is None:
        axes = tuple(reversed(range(a.data.ndim)))
    out = Tensor(
        np.transpose(a.data, axes),
        requires_grad=a.requires_grad,
        _op='transpose',
        _children=(a,)
    )
    
    def _backward():
        if a.requires_grad:
            inv_axes = tuple(np.argsort(axes))
            a.grad += np.transpose(out.grad, inv_axes)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def reshape(a, shape):
    """Reshape tensor"""
    a = _ensure_tensor(a)
    original_shape = a.data.shape
    out = Tensor(
        np.reshape(a.data, shape),
        requires_grad=a.requires_grad,
        _op='reshape',
        _children=(a,)
    )
    
    def _backward():
        if a.requires_grad:
            a.grad += np.reshape(out.grad, original_shape)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out

