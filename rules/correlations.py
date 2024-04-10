import pandas as pd
import numpy as np
from scipy import spatial

def C(input,n):
    
    input = input.squeeze(0)
    
    print(np.shape(input))
    
    a = np.zeros([n,n])
    for i in range(n) :
      for j  in range(n) :
        if j>i : 
          a[i,j] = a[j,i]
        if j == i :
          a[i,j] = 0
        else :
          a[i,j] = 1 - spatial.distance.cosine(input[:,i],input[:,j])
    return a