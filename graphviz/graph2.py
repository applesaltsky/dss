import graphviz

g = graphviz.Graph('G', filename='fdpclust.gv', engine='fdp')

g.node('e')

with g.subgraph(name='clusterA') as a:
    a.node('a')
    a.node('b')
    a.edge('a', 'b',style='invis')

with g.subgraph(name='clusterB') as b:
    b.edge('d', 'f',style='invis')

#g.edge('d', 'D')
g.edge('e', 'clusterA')
g.edge('clusterA', 'clusterB')

g.view(cleanup=True)
print(g.source)