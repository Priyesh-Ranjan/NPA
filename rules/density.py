import torch
import torch.nn as nn

import numpy as np
from rules.correlations import C 

import math
def density(r) :
  sum = 0
  for i in range(len(r[0])) :
    sum += np.sum(r[i])
  return sum/(len(r[0]))

def fun(input) :
    n = input.shape[-1]
    r = C(input,n)
    d = 0
    h = []
    alt = []
    i = 0
    for k in range(n - 1) :
        o = density(r)
        temp = np.delete(r,i,0)
        l = np.delete(temp,i,1)
        d = density(l)
        if o < d :
            r = l
            i-=1
            h.append(k)
        i+=1
    input = input.squeeze(0)
    r2 = C(input,n)
    alt = ([r2[i][j] for i in range(n) for j in range(n) if i in h and j in h and i != j])
    m = ([r2[i][j] for i in range(n) for j in range(n) if i not in h and j not in h and i != j])
    avg = np.sum(alt)/len(h)
    #avg = np.median(alt)
    den = np.median(m)
    if len(h) < 2:
        out = torch.mean(input[:,[i for i in range(n) if i not in h]], dim=1, keepdim=True)
        print(h)
        return out
    elif len(h) > n-2:
        out = torch.mean(input[:,[i for i in range(n) if i in h]], dim=1, keepdim=True)
        print([i for i in range(n) if i not in h])
        return out
    elif den < avg :
        out = torch.mean(input[:,[i for i in range(n) if i not in h]], dim=1, keepdim=True)
        print(h)
        print(avg)
        print(den)
        return out
    elif den > avg :
        out = torch.mean(input[:,[i for i in range(n) if i in h]], dim=1, keepdim=True)
        print([i for i in range(n) if i not in h])
        print(den)
        print(avg)
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