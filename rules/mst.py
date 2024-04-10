import torch
import torch.nn as nn
import networkx as nx

import numpy as np
from rules.correlations import C 

def find(i,parent):
	while parent[i] != i:
		i = parent[i]
	return i
	
def union(i, j,parent):
	a = find(i,parent)
	b = find(j,parent)
	parent[a] = b
	return parent
	
def fun(input):
    n = input.shape[-1]
    parent = [i for i in range(n)]
    maxcost = 0
    INF = float('inf')
    edge_count = 0;G = nx.DiGraph();
    G.add_nodes_from([i for i in range(n)])
    cost = C(input,n)
    while edge_count < n - 2:
        max = -1* INF
        a = -1
        b = -1
        for i in range(n):
            for j in range(n):
                if find(i,parent) != find(j,parent) and cost[i][j] > max:
                    max = cost[i][j]
                    a = i
                    b = j
        parent = union(a, b, parent)
        G.add_edge(a,b)
        edge_count += 1
        maxcost += max

    UG = G.to_undirected();
    sub_graphs = [UG.subgraph(c) for c in nx.connected_components(UG)]
    min_d = -1*n
    p = []
    k = [[],[]]
    for i, sg in enumerate(sub_graphs) :
        k = [int(j) for j in sg.nodes]
        f = [j for j in sg.edges.data("weight")]
        #print(f)
        #print([cost[x[0]] for x in f])
        if len(k) < 2 :
            p = k
            break
        if len(k) > n-2 :
            p = [j for j in range(n) if j not in k]
            break
        d = np.average([np.sum(cost[x[0]]) for x in f])
        #d = np.median([cost[x[0]][x[1]] for x in f])
        print(d)
        print(k)
        if d > min_d :
            min_d = d
            p = k
    input = input.squeeze(0)        
    out = torch.mean(input[:,[i for i in range(n) if i not in p]], dim=1, keepdim=True)
    print(p)
    return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, input):
        #         print(input.shape)
        '''
        input: batchsize* vector dimension * n 
        (1 by d by n)
        
        return 
            out : size =vector dimension, will be flattened afterwards
        '''
        out = fun(input)

        return out