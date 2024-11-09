#https://www.graphviz.org/docs/attrs/arrowtail/
#dot -Tpng C:\Users\User\Desktop\Programming\dss\graph\first.gv > output.png

import graphviz
from pathlib import Path
import io

PATH_PROJECT = Path(__file__).parent.parent.parent

engine = 'dot'  #DEFAULT ENGINE : dot
graph = graphviz.Digraph(engine=engine) 
DIRECTION = 'LR' #'TB'
TITLE = 'TEST GRAPH'
graph.attr(**{'rankdir':DIRECTION,'compound':'true','label':TITLE,'labelloc':'t', 'center':'true','rank':'same'})


#set node
with graph.subgraph(name='input') as i:
    i.node_attr = {'shape':'circle'}
    i.graph_attr = {'cluster':'true','label':'input'}
    i.node('c', '1', style='filled', fillcolor='lightgray')
    i.node('i1', 'x1')  
    i.node('i2', 'x2')

with graph.subgraph(name='hidden1') as h1:
    #SHAPE : https://www.graphviz.org/doc/info/shapes.html
    h1.node_attr = {'shape':'circle'}
    h1.graph_attr = {'cluster':'true','label':'hidden1'}
    h1.node('h1c', '1',style='filled',fillcolor='lightgray')

    with h1.subgraph(name='h1n1') as h1n1:  
        h1n1.graph_attr = {'label':''}
        h1n1.node('h1n1a', 'a1')
        h1n1.node('h1n1z', 'z1')
        h1n1.edge('h1n1a','h1n1z',label='h()')

    with h1.subgraph(name='h1n2') as h1n2:    
        h1n2.graph_attr = {'label':''}
        h1n2.node('h1n2a', 'a2')
        h1n2.node('h1n2z', 'z2')
        h1n2.edge('h1n2a','h1n2z',label='h()')

    with h1.subgraph(name='h1n3') as h1n3:    
        h1n3.graph_attr = {'label':''}
        h1n3.node('h1n3a', 'a3')
        h1n3.node('h1n3z', 'z3')
        h1n3.edge('h1n3a','h1n3z',label='h()')    

with graph.subgraph(name='h2') as h2:
    #SHAPE : https://www.graphviz.org/doc/info/shapes.html
    h2.node_attr = {'shape':'circle'}
    h2.graph_attr = {'cluster':'true','label':'hidden2'}
    h2.node('h2c', '1',style='filled',fillcolor='lightgray')

    with h2.subgraph(name='h2n1') as h2n1:  
        h2n1.graph_attr = {'label':''}
        h2n1.node('h2n1a', 'a1')
        h2n1.node('h2n1z', 'z1')
        h2n1.edge('h2n1a','h2n1z',label='h()')

    with h2.subgraph(name='h2n2') as h2n2:    
        h2n2.graph_attr = {'label':''}
        h2n2.node('h2n2a', 'a2')
        h2n2.node('h2n2z', 'z2')
        h2n2.edge('h2n2a','h2n2z',label='h()')

with graph.subgraph(name='out') as o:
    o.node_attr = {'shape':'circle'}
    o.graph_attr = {'cluster':'true','label':'output'}
    o.node('o1', 'y1')
    o.node('o2', 'y2')


#set edges
for i in ['c','i1','i2']:
    for o in ['h1n1a','h1n2a','h1n3a']:
        graph.edge(i,o,label='b' if i == 'c' else 'w')

for i in ['h1c','h1n1z','h1n2z','h1n3z']:
    for o in ['h2n1a','h2n2a']:
        graph.edge(i,o,label='b' if i == 'h1c' else 'w')

for i in ['h2c','h2n1z','h2n2z']:
    for o in ['o1','o2']:
        graph.edge(i,o,label='b' if i == 'h2c' else 'w')

#render
graph.render(engine=engine, outfile=Path(PATH_PROJECT,'graph','ch3.png'), format='png', view=True)
print(graph.source)