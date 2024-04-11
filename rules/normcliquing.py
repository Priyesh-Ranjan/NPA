import torch
import torch.nn as nn

import numpy as np
import sklearn.metrics.pairwise as smp

def bronk(R, P, X, g):
    if not any((P, X)):
        yield R
    for v in P[:]:
        R_v = R + [v]
        P_v = [v1 for v1 in P if v1 in N(v, g)]
        X_v = [v1 for v1 in X if v1 in N(v, g)]
        for r in bronk(R_v, P_v, X_v, g):
            yield r
        P.remove(v)
        X.append(v)
def N(v, g):
    return [i for i, n_v in enumerate(g[v]) if n_v]

class Net(nn.Module):
    def __init__(self, eps = 0.05, gamma = 0.5, kappa = 2, tau = 0.35, n_clients = 50, init_norm = 0):
        super(Net, self).__init__()
        self.kappa = kappa
        self.tau = tau
        self.norm = kappa*init_norm
        self.gamma = gamma
        self.eps = eps
        self.reputation = np.ones(n_clients)
        self.delta = 0.1
        self.n_clients = n_clients

    def forward(self, input):
        #         print(input.shape)
        '''
        input: batchsize* vector dimension * n 
        (1 by d by n)
        
        return 
            out : size =vector dimension, will be flattened afterwards
        '''
        out = self.adaptor(input)

        return out
    
    def adaptor(self, deltas):
        '''
        compute foolsgold 
        
        input : 1* vector dimension * n
        
        return 
            foolsGold :  vector dimension 
        '''
        x = deltas.squeeze(0)
        x = x.permute(1, 0)
        w = self.main(x)
        print(w)
        w = w / w.sum()
        out = torch.sum(x.permute(1, 0) * w, dim=1, keepdim=True)
        return out
    
    def main(self, grads):
        n_clients = grads.shape[0]
                
        # Initializing Norm-Clipping
                
        norms = [grad.norm(p=2) for grad in grads]
        idx = [1 if norm <= self.norm else 0 for norm in norms]
        if sum(idx)/len(idx) > self.tau :
            self.norm = np.percentile(norms, self.tau*100)
        grads = [torch.div(grad, max(1,grad.norm(p=2)/self.norm)) for grad in grads]
        grads = torch.stack(grads, 1).permute(1,0)        
        # Finding Cliques
                
        Honest = []
        cs = smp.euclidean_distances(grads)
        neighbors = np.zeros_like(cs)
        
        gamma = self.gamma
        
        while len(Honest) == 0 :
            for i in range(n_clients) :
                for j in range(i+1,n_clients) :
                    if cs[i,j] < gamma*self.reputation[i] : 
                        neighbors[i,j] = 1
                        neighbors[j,i] = 1
            Cliques = list(bronk([], [*range(n_clients)], [], neighbors))  
            if len(max(Cliques, key=len)) > n_clients/2 :
                Honest = max(Cliques, key=len)
                print(Honest)
                break
            gamma += self.eps
        
        wv = np.array([1 if i in Honest else 0 for i in range(n_clients)])    
        
        self.reputation = np.array([min(1,i+self.delta) if i in Honest else max(0,i-self.delta) for i in self.reputation])
        
        return wv
