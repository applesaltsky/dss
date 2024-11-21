from pathlib import Path
import sys
PATH_PY = Path(__file__)
PATH_PROJECT = PATH_PY.parent.parent
sys.path.append(PATH_PROJECT.__str__())

from typing import Callable,Union
import pickle
import util
import numpy as np
import graphviz
from itertools import pairwise

class Network:
    def __init__(self):
        self.layers = []

    def add_layers(self,input_cnt:int,output_cnt:int,activation_func:Callable,seed:Union[None,int] = None):
        #check input
        if seed is not None:
            if not isinstance(seed,int):
                raise ValueError('seed should be integer')
            np.random.seed(seed)

        if not (isinstance(input_cnt,int) and isinstance(output_cnt, int)):
            raise ValueError('input val and output val must be positive integer')
        
        if not (input_cnt > 0 and output_cnt > 0):
            raise ValueError('input val and output val must be positive integer')

        if len(self.layers) > 0:
            last_ouput_cnt = self.layers[-1]['output_cnt']
            if not (last_ouput_cnt == input_cnt):
                raise ValueError(f'input val should be eqult to output cnt of last layer. {last_ouput_cnt}.')

        #to do : define layer class...  
        self.layers.append({'input_cnt':input_cnt,
                            'output_cnt':output_cnt,
                            'activation_func':activation_func(),
                            'activation_name':activation_func.__name__,
                            'weight':np.random.randn(input_cnt,output_cnt),
                            'bias':np.random.randn(output_cnt),
                            'affine':None,
                            'output':None,
                            })
        
        if seed is not None:
            np.random.seed()

    def predict(self,input_val:np.ndarray,verbose:bool=False):
        #check input
        if not isinstance(input_val,np.ndarray):
            raise ValueError('input should be np.ndarray.')

        #if not (input_val.ndim == 1):
        #    raise ValueError(f'input ndim should be 1.')
        
        if not (len(self.layers) > 2):
            raise IndexError('The list of layers must be at least 2 in length.')

        input_len = self.layers[0]['input_cnt']
        if not (input_len == input_val.shape[-1]):
            raise ValueError(f'input length should be {input_len}.')
        
        #prediction
        temp = input_val
        if verbose:
            print(input_val)
        for layer in self.layers:
            layer['affine'] = np.dot(temp, layer['weight']) + layer['bias']
            layer['output'] = layer['activation_func'](layer['affine'])
            temp = layer['output']
            if verbose:
                print(temp)

        return temp
    
    def draw(self,filepath:Union[str,Path,None]=None,view:bool=True)->str:
        if not (len(self.layers) > 2):
            raise IndexError(f"The list of layers must be at least 2 in length.")
        
        ENGINE = 'dot'
        DIRECTION = 'LR'
        MAX_NODE_VIEW = 3
        
        #set save filepath
        filepath = Path(Path(__file__).parent,'out.png').__str__() if filepath is None else filepath.__str__()

        #set graphviz config
        graph = graphviz.Digraph(engine=ENGINE)
        graph.attr(**{'rankdir':DIRECTION,'compound':'true','labelloc':'t', 'center':'true','rank':'same'})

        #set node config
        nodes = []
        node_cnt = self.layers[0]['input_cnt']
        node_real_cnt = node_cnt
        node_cnt = MAX_NODE_VIEW if node_cnt >= MAX_NODE_VIEW else node_cnt
        node_name = 'input'
        nodes.append({'node_name':node_name,'node_cnt':node_cnt,'node_activation':'','node_real_cnt':node_real_cnt})
        for idx, layer in enumerate(self.layers):
            node_cnt = layer['output_cnt']
            node_real_cnt = node_cnt
            node_cnt = MAX_NODE_VIEW if node_cnt >= MAX_NODE_VIEW else node_cnt
            node_activation = layer['activation_name']
            node_name = f'hidden{idx+1}'
            nodes.append({'node_name':node_name,'node_cnt':node_cnt,'node_activation':node_activation,'node_real_cnt':node_real_cnt})
        nodes[-1]['node_name'] = 'output'
        #node : [{'node_name': 'input', 'node_cnt': 2, 'node_activation': ''}, {'node_name': 'hidden1', 'node_cnt': 3, 'node_activation': 'sigmoid'}, {'node_name': 'hidden2', 'node_cnt': 2, 'node_activation': 'sigmoid'}, {'node_name': 'output', 'node_cnt': 2, 'node_activation': 'identity_function'}]
        
        #set node
        for node in nodes:
            node_name = node['node_name']
            node_activation = node['node_activation'] 
            node_cnt = node['node_cnt']
            node_real_cnt = node['node_real_cnt']
            with graph.subgraph(name=node_name) as n:
                n.node_attr = {'shape':'circle'}
                
                if node_name == 'input':
                    n.graph_attr = {'cluster':'true','label':f'{node_name}_{node_real_cnt}'}
                    n.node('ic', '1', style='filled', fillcolor='lightgray')
                    for ni in range(node_cnt):
                        n.node(f'i{ni}', f'x{ni}') 

                elif 'hidden' in node_name:
                    n.graph_attr = {'cluster':'true','label':f'{node_activation}_{node_real_cnt}'}
                    n.node(f'{node_name}c', '1',style='filled',fillcolor='lightgray')
                    for ni in range(node_cnt):
                        with n.subgraph(name=f'{node_name}n1') as nsub:  
                            nsub.graph_attr = {'label':''}
                            node_input = f'{node_name}n{ni}a'
                            node_output = f'{node_name}n{ni}z'
                            nsub.node(node_input, f'a{ni}')
                            nsub.node(node_output, f'z{ni}')
                            nsub.edge(node_input,node_output,label='h()')   

                elif node_name == 'output':
                    n.graph_attr = {'cluster':'true','label':f'{node_activation}_{node_real_cnt}'}
                    for ni in range(node_cnt):
                        n.node(f'ao{ni}',f'a{ni}')
                        n.node(f'o{ni}',f'y{ni}')
                        n.edge(f'ao{ni}',f'o{ni}',label='h()')

                else:
                    pass

        #set edge config
        for i,o in pairwise(nodes):
            #{'node_name': 'input', 'node_cnt': 2} {'node_name': 'hidden1', 'node_cnt': 3}
            #{'node_name': 'hidden2', 'node_cnt': 2} {'node_name': 'output', 'node_cnt': 2}

            input_node_cnt = i['node_cnt']
            input_node_name = i['node_name']
            output_node_cnt = o['node_cnt']
            output_node_name = o['node_name']

            if input_node_name == 'input':
                edge_start = ['ic'] + [f'i{ni}' for ni in range(input_node_cnt)]
                edge_end = [f'{output_node_name}n{ni}a' for ni in range(output_node_cnt)]
  
            elif output_node_name == 'output':
                edge_start = [f'{input_node_name}c'] + [f'{input_node_name}n{ni}z' for ni in range(input_node_cnt)]
                edge_end = [f'ao{ni}' for ni in range(output_node_cnt)]
            else:
                edge_start = [f'{input_node_name}c'] + [f'{input_node_name}n{ni}z' for ni in range(input_node_cnt)]
                edge_end = [f'{output_node_name}n{ni}a' for ni in range(output_node_cnt)]
            
            for e_s in edge_start:
                for e_e in edge_end:
                    graph.edge(e_s,e_e)
                
        #draw graph
        graph.render(engine=ENGINE, outfile=filepath, format='png', view=view)
        
        return graph.source
    
    def dump(self,filepath=None):
        filepath = Path(Path(__file__).parent,'network.pickle').__str__() if filepath is None else filepath.__str__()
        with open(filepath,'wb') as f:
            pickle.dump(self,f)

    def load(self,filepath=None):
        filepath = Path(Path(__file__).parent,'network.pickle').__str__() if filepath is None else filepath.__str__()
        with open(filepath,'rb') as f:
            self = pickle.load(f)
        return self

network = Network()
#network = network.load()
network.add_layers(*(784,50,util.activation_func.sigmoid))          
network.add_layers(*(50,100,util.activation_func.sigmoid)) 
network.add_layers(*(100,10,util.activation_func.softmax))



X, y = util.datasets.load_mnist()
#input_val = X[0] / np.max(X[0])
#predict_val = network.predict(input_val,verbose=False)
#print(predict_val)

#network.dump()
#print(network.draw())

predict_val = network.predict(X/np.max(X),verbose=False)
eval_val = []
for _y in y:
    t = [0]*10
    t[int(_y)] = 1
    eval_val.append(t)
eval_val = np.array(eval_val)

accuracy = util.eval_func.accuracy(predict_val,eval_val)
print(accuracy)
