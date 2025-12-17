import numpy as np
from autograd import Tensor


class SGD:
    """Stochastic Gradient Descent with momentum"""
    
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        """
        Args:
            parameters: list of Tensor parameters to optimize
            lr: learning rate
            momentum: momentum coefficient (0.9 as required)
        """
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}
        
        # Initialize velocity for each parameter
        for param in self.parameters:
            self.velocity[id(param)] = np.zeros_like(param.data)
    
    def step(self):
        """Perform one optimization step"""
        for param in self.parameters:
            if param.grad is None:
                continue
            
            param_id = id(param)
            
            # Update velocity with momentum
            self.velocity[param_id] = (
                self.momentum * self.velocity[param_id] - self.lr * param.grad
            )
            
            # Update parameter
            param.data += self.velocity[param_id]
    
    def zero_grad(self):
        """Reset gradients to zero"""
        for param in self.parameters:
            if param.grad is not None:
                param.grad.fill(0)

