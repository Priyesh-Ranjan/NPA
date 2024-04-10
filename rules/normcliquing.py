import torch
import torch.nn as nn

import numpy as np
import sklearn.metrics.pairwise as smp


# Takes in grad
# Compute similarity
# Get weightings
def foolsgold(grads):
    n_clients = grads.shape[0]
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    
    

    return wv


def adaptor(input):
    '''
    compute foolsgold 
    
    input : 1* vector dimension * n
    
    return 
        foolsGold :  vector dimension 
    '''
    x = input.squeeze(0)
    x = x.permute(1, 0)
    w = foolsgold(x)
    print(w)
    w = w / w.sum()
    out = torch.sum(x.permute(1, 0) * w, dim=1, keepdim=True)
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
        out = adaptor(input)
        print(type(out))

        return out
