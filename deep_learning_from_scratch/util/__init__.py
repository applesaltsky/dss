import numpy as np
from typing import Union

class activation_func:
    #for hidden layer
    def step_func(x:Union[np.ndarray,int,float])->int:
        if isinstance(x,np.ndarray):
            y = x > 0
            return y.astype(np.int)
        else:
            return 1 if x > 0 else 0

    def sigmoid(x:Union[np.ndarray,int,float]):
        return 1 / (1+np.exp(-x))
    
    def relu(x:Union[np.ndarray,int,float]):
        return np.maximum(0,x)
    
    #for output
    def identity_function(x):
        return x
    
    def softmax(x):
        c = np.max(x)  #prevent overflow
        return np.exp(x - c)/np.sum(np.exp(x - c))