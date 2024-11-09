#https://www.graphviz.org/docs/attrs/arrowtail/

import graphviz
from pathlib import Path
import io

PATH_PROJECT = Path(__file__).parent.parent

engine = 'dot'  #DEFAULT ENGINE : dot
graph = graphviz.Digraph(engine=engine) 
DIRECTION = 'LR' #'TB'
TITLE = 'TEST GRAPH'
graph.attr(**{'rankdir':DIRECTION,'compound':'true','label':TITLE,'labelloc':'t'})

#set node
graph.node('START','START',shape='Mdiamond')
graph.node('A', 'PYTHON')  
graph.node('B', 'JAVASCRIPT')
graph.node('C', 'JAVA')

with graph.subgraph(name='GR1') as g1:
    #SHAPE : https://www.graphviz.org/doc/info/shapes.html
    g1.node_attr = {'shape':'box','style':'filled','color':'green','fillcolor':'lightgray','label':'group1','label':'gr1','fontcolor':'black'}
    g1.graph_attr = {'cluster':'true','label':'low level language'}
    g1.node('D', 'RUST')
    g1.node('E', 'C++')
    g1.node('F', 'C')

graph.node('G', 'GO')
graph.node('END','END',shape='Mdiamond')


#set edges
graph.edge('START','A') 
graph.edge('A','B',label='A to B')     #A->B
graph.edge('B','C')                    #B->C
graph.edge('C','E',lhead='GR1')        #C->GR1
graph.edge('E','G',ltail='GR1')        #GR1->G
graph.edge('G','END', shape='Msquare') #G->END

#render
graph.render(engine=engine, outfile=Path(PATH_PROJECT,'graph','first.png'), format='png', view=True)
print(graph.source)