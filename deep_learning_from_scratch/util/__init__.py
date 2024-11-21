import numpy as np
from typing import Union
from sklearn.datasets import fetch_openml

class trainer:
    def numerical_diff(self, f,x):
        h = 1e-4
        return (f(x+h) - f(x-h))/(2*h)

    def numerical_gradient(self, f, x):
        h = 1e-4
        grad = np.zeros_like(x)
        #print(x.shape)
        for idx in range(x.shape[0]):
            for idx2 in range(x.shape[1]):
                tmp_val = x[idx,idx2]
                
                x[idx,idx2] = tmp_val + h
                fxh1 = f(x)
        
                x[idx,idx2] = tmp_val - h
                fxh2 = f(x)
        
                grad[idx,idx2] = (fxh1 - fxh2)/(2*h)
                x[idx,idx2] = tmp_val
        
        return grad
    
class activation_func:
    #for hidden layer
    def step_func(x:Union[np.ndarray,int,float])->int:
        if isinstance(x,np.ndarray):
            y = x > 0
            return y.astype(np.int)
        else:
            return 1 if x > 0 else 0

    class sigmoid:
        def __call__(self,x:Union[np.ndarray,int,float]):
            return 1 / (1+np.exp(-1*x))
        
        def backward(self,x):
            return self.__call__(x) * (1 - self.__call__(x))
        
    class relu:
        def __call__(self,x:Union[np.ndarray,int,float]):
            return np.maximum(0,x)
    
    class leaky_relu:
        def __call__(self,x:Union[np.ndarray,int,float]):
            return np.maximum(0.01*x,x)
    
    #for output
    class identity_function:
        def __call__(self,x):
            return x
    
    #for classification output
    class softmax:
        def __call__(self,x):
            c = np.max(x)  #prevent overflow
            return np.exp(x - c)/np.sum(np.exp(x - c))
        
        def backward(self,x):
            identity_backward = self.__call__(x) * ( 1- self.__call__(x) )     
            identity_mtx = identity_backward * np.identity(x.shape[0],dtype=float) 

            xv, yv = np.meshgrid(x, x)
            non_identity_backward = -1 * self.__call__(xv) * self.__call__(yv)
            non_identity_mtx =  non_identity_backward * (np.identity(x.shape[0]) != 1).astype(float)

            return identity_mtx + non_identity_mtx
            
            
    

class loss_func:   
    #for clasification
    class cross_entropy_error:
        def __call__(self, predict_val:np.ndarray, eval_val:np.ndarray)->float:
            if predict_val.ndim == 1:
                predict_val = predict_val.reshape(1,predict_val.size)
                eval_val = eval_val.reshape(1,eval_val.size)
            
            batch_size = predict_val.shape[0]
            delta = 1e-7
            return -np.sum(eval_val * np.log(predict_val + delta)) / batch_size
        
        def backward(self, predict_val, eval_val):
            return -1 * (eval_val / predict_val)

    #for regression
    def sum_squares_error(predict_val:np.ndarray, eval_val:np.ndarray)->float:
        return np.sum((predict_val-eval_val)**2)/2
    

class eval_func:     
    #for 
    def accuracy(predict_val:np.ndarray,eval_val:np.ndarray)->float:
        '''
        predict [[0,0.97,0.03],[15.,44.4,40.6],[0.8,0,0.2],[0,0.1,0.9]]   
        eval    [[0,1,0],[1,0,0],[1,0,0],[0,0,1]]    

        '''
        tmp = (np.argmax(predict_val, axis=1) == np.argmax(eval_val,axis=1))
        return np.sum(tmp)/tmp.shape[0]    

class datasets:
    def load_mnist():
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        return X, y
    
if __name__ == "__main__":
    t = activation_func.sigmoid()
    print(t(3))
    print(t.backward(3))
    print(t.__class__.__name__)