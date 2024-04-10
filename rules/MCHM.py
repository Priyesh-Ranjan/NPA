import torch
import torch.nn as nn

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import utils
import time


def fun(input) :
    
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(np.array(input.squeeze(0)))
    epsilon = 0.05
    
    for i in range(10) :
    
        for index,ele in enumerate(arr) :
    
            p = ele - np.mean(ele)
            arr[index] = ele*(1.0 - abs(p)*np.sign(p)) + 0.001*int(i==0)
        
        epsilon /= 10    
    
    out = torch.tensor(scaler.inverse_transform(arr), dtype = torch.float64).view(1,-1,len(ele))
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