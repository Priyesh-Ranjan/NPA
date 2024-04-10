import torch
import torch.nn as nn

import numpy as np
from utils import utils


def fun(input) :
       
    out = torch.tensor(input, dtype = torch.float64)
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