import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self, 
                              x: List[float], 
                              W1: List[List[float]], b1: List[float], 
                              W2: List[List[float]], b2: List[float], 
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        
        ## forward
        W1 = np.array(W1)
        W2 = np.array(W2)
        x = np.array(x)
        b1 = np.array(b1)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        
        z1_pre = W1 @ x + b1
        a1 = np.maximum(0, z1_pre)
        y = W2 @ a1 + b2
        
        ## loss
        loss = np.mean((y - y_true)**2)

        dL_dy = 2/len(y) * (y - y_true)
        dL_dW2 = np.outer(dL_dy, a1)
        dL_db2 = dL_dy

        dL_da1 = dL_dy @ W2
        dL_dz1 = dL_da1 * (z1_pre > 0)

        dL_dW1 = np.outer(dL_dz1, x)
        dL_db1 = dL_dz1

        return {
            'loss': float(np.round(loss, 4)),
            'dW1': np.round(dL_dW1, 4).tolist(),
            'db1': np.round(dL_db1, 4).tolist(),
            'dW2': np.round(dL_dW2, 4).tolist(),
            'db2': np.round(dL_db2, 4).tolist()
        }