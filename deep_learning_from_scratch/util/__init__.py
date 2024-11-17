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
    
    def gradient_descent(self, f, init_w, lr=0.01, step_num = 100):
        w = init_w
        his = []
        for i in range(step_num):
            his.append(w.copy())
            grad = self.numerical_gradient(f, w)
            w -= lr * grad
        return his

class activation_func:
    #for hidden layer
    def step_func(x:Union[np.ndarray,int,float])->int:
        if isinstance(x,np.ndarray):
            y = x > 0
            return y.astype(np.int)
        else:
            return 1 if x > 0 else 0

    def sigmoid(x:Union[np.ndarray,int,float]):
        return 1 / (1+np.exp(-1*x))
    
    def relu(x:Union[np.ndarray,int,float]):
        return np.maximum(0,x)
    
    def leaky_relu(x:Union[np.ndarray,int,float]):
        return np.maximum(0.01*x,x)
    
    #for output
    def identity_function(x):
        return x
    
    #for classification output
    def softmax(x):
        c = np.max(x)  #prevent overflow
        return np.exp(x - c)/np.sum(np.exp(x - c))
    

class loss_func:   
    #for clasification
    def cross_entropy_error(predict_val:np.ndarray, eval_val:np.ndarray)->float:
        if predict_val.ndim == 1:
            predict_val = predict_val.reshape(1,predict_val.size)
            eval_val = eval_val.reshape(1,eval_val.size)
        
        batch_size = predict_val.shape[0]
        delta = 1e-7
        return -np.sum(eval_val * np.log(predict_val + delta)) / batch_size

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