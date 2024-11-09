from pathlib import Path
import sys
PATH_PY = Path(__file__)
PATH_PROJECT = PATH_PY.parent.parent
sys.path.append(PATH_PROJECT.__str__())

from typing import Callable,Union
import util
import numpy as np
import graphviz
from itertools import pairwise

class Network:
    def __init__(self):
        self.layers = []

    def add_layers(self,input_cnt:int,output_cnt:int,activation_func:Callable,seed:Union[None,int] = None):
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

        self.layers.append({'input_cnt':input_cnt,
                            'output_cnt':output_cnt,
                            'activation_func':activation_func,
                            'activation_name':activation_func.__name__,
                            'weight':np.random.randn(input_cnt,output_cnt),
                            'bias':np.random.randn(output_cnt)
                            })
        
        if seed is not None:
            np.random.seed()

    def predict(self,input_val:np.ndarray,verbose:bool=False):
        #check input
        if not isinstance(input_val,np.ndarray):
            raise ValueError('input should be np.ndarray.')

        if not (input_val.ndim == 1):
            raise ValueError(f'input ndim should be 1.')

        input_len = self.layers[0]['input_cnt']
        if not (input_len == len(input_val)):
            raise ValueError(f'input length should be ({input_len}).')
        
        #prediction
        temp = input_val
        if verbose:
            print(input_val)
        for layer in self.layers:
            a = np.dot(temp, layer['weight']) + layer['bias']
            z = layer['activation_func'](a)
            temp = z
            if verbose:
                print(temp)

        return temp
    
    def draw(self,filepath:Union[str,Path,None]='out.png',view:bool=True)->str:
        if not (len(self.layers) > 2):
            raise IndexError(f"The list of layers must be at least 2 in length.")
        
        #set save filepath
        filepath = Path(Path(__file__).parent,filepath).__str__()

        #set graphviz config
        engine = 'dot'
        graph = graphviz.Digraph(engine=engine)
        DIRECTION = 'LR'
        graph.attr(**{'rankdir':DIRECTION,'compound':'true','labelloc':'t', 'center':'true','rank':'same'})

        #set node config
        nodes = []
        node_cnt = self.layers[0]['input_cnt']
        node_name = 'input'
        nodes.append({'node_name':node_name,'node_cnt':node_cnt,'node_activation':''})
        for idx, layer in enumerate(self.layers):
            node_cnt = layer['output_cnt']
            node_activation = layer['activation_name']
            node_name = f'hidden{idx+1}'
            nodes.append({'node_name':node_name,'node_cnt':node_cnt,'node_activation':node_activation})
        nodes[-1]['node_name'] = 'output'
        #node : [{'node_name': 'input', 'node_cnt': 2, 'node_activation': ''}, {'node_name': 'hidden1', 'node_cnt': 3, 'node_activation': 'sigmoid'}, {'node_name': 'hidden2', 'node_cnt': 2, 'node_activation': 'sigmoid'}, {'node_name': 'output', 'node_cnt': 2, 'node_activation': 'identity_function'}]
        
        #set node
        for node in nodes:
            node_name = node['node_name']
            node_activation = node['node_activation'] 
            node_cnt = node['node_cnt']
            with graph.subgraph(name=node_name) as n:
                n.node_attr = {'shape':'circle'}
                
                if node_name == 'input':
                    n.graph_attr = {'cluster':'true','label':node_name}
                    n.node('ic', '1', style='filled', fillcolor='lightgray')
                    for ni in range(node_cnt):
                        n.node(f'i{ni}', f'x{ni}') 

                elif 'hidden' in node_name:
                    n.graph_attr = {'cluster':'true','label':node_activation}
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
                    n.graph_attr = {'cluster':'true','label':node_activation}
                    for ni in range(node_cnt):
                        n.node(f'o{ni}',f'y{ni}')

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
                edge_end = [f'o{ni}' for ni in range(output_node_cnt)]
            else:
                edge_start = [f'{input_node_name}c'] + [f'{input_node_name}n{ni}z' for ni in range(input_node_cnt)]
                edge_end = [f'{output_node_name}n{ni}a' for ni in range(output_node_cnt)]
            
            for e_s in edge_start:
                for e_e in edge_end:
                    graph.edge(e_s,e_e)
                

        #draw graph
        graph.render(engine=engine, outfile=filepath, format='png', view=view)
        
        return graph.source

network = Network()
network.add_layers(*(2,4,util.activation_func.sigmoid))        
network.add_layers(*(4,3,util.activation_func.sigmoid))   
network.add_layers(*(3,2,util.activation_func.relu)) 
network.add_layers(*(2,2,util.activation_func.identity_function))

#input_val = np.array([1,2])
#predict_val = network.predict(input_val,verbose=True)
#print(predict_val)
print(network.draw())


