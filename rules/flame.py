import torch
import torch.nn as nn

import numpy as np
import statistics as sts
import sklearn.metrics.pairwise as smp


def fun(grads) :
  n_clients = grads.shape[0]
  cosine_vals = smp.cosine_similarity(grads) - np.eye(n_clients)
  
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
        x = input.squeeze(0)
        x = x.permute(1, 0)
        out = fun(input)

        return out