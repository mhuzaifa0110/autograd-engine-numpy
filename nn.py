import numpy as np
from autograd import Tensor
import math


class Module:
    """Base class for all neural network modules."""
    
    def __init__(self):
        self.parameters = []
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, x):
        raise NotImplementedError
    
    def train(self):
        """Set module to training mode."""
        pass
    
    def eval(self):
        """Set module to evaluation mode."""
        pass


class Linear(Module):
    """Fully connected layer: y = xW^T + b"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier/Glorot initialization
        bound = math.sqrt(6.0 / (in_features + out_features))
        weight_data = np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32)
        self.weight = Tensor(weight_data, requires_grad=True)
        
        if bias:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
        else:
            self.bias = None
        
        self.parameters = [self.weight]
        if self.bias is not None:
            self.parameters.append(self.bias)
    
    def forward(self, x):
        # x: (batch_size, in_features)
        # weight: (out_features, in_features)
        # We need to transpose weight for matmul: x @ W^T
        out = x @ self.weight.transpose((1, 0))
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLU(Module):
    """ReLU activation function"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    """Tanh activation function"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    """Sigmoid activation function"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.sigmoid()


class Sequential(Module):
    """Container for sequential layers"""
    
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        self.parameters = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                self.parameters.extend(layer.parameters)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def cross_entropy_loss(logits, targets):
    """
    Cross-entropy loss with numerical stability.
    logits: (batch_size, num_classes)
    targets: (batch_size,) integer class labels
    """
    batch_size = logits.data.shape[0]
    
    # Numerical stability: subtract max
    max_logits = logits.max(axis=1, keepdims=True)
    logits_stable = logits - max_logits
    
    # Compute softmax
    exp_logits = logits_stable.exp()
    sum_exp = exp_logits.sum(axis=1, keepdims=True)
    probs = exp_logits / sum_exp
    
    # Compute cross-entropy
    log_probs = logits_stable - sum_exp.log()
    
    # Gather correct class log probabilities
    targets_one_hot = np.zeros_like(logits.data)
    targets_one_hot[np.arange(batch_size), targets] = 1.0
    targets_tensor = Tensor(targets_one_hot, requires_grad=False)
    
    loss = -(log_probs * targets_tensor).sum() / batch_size
    
    return loss, probs



