import torch
import torch.nn as nn
#import networkx as nx
from copy import deepcopy

import numpy as np
#from rules.correlations import C 
from utils import convert_pca, utils
import sklearn.metrics.pairwise as smp


	
def fun(grads):
    
    """#print(stacked)
    
    val = [input for input in stacked['fc3.weight'].numpy()]
    bias = [input for input in stacked['fc3.bias'].numpy()]
    
    #print(np.shape(val))
    #print(np.shape(bias))
    
    arr = np.reshape(val,(50,10,84))
    bs = np.transpose(bias)"""
    
    n_clients = grads.shape[0]
    
    cs = smp.euclidean_distances(grads)
    maxcs = np.argsort(cs, axis = -1)
    
    w = np.zeros((n_clients,1))
    
    for i in range(n_clients) :
        w[i] += np.mean([1 - (2*np.where(row == i)[0])/(n_clients-1) for row in maxcs])
        #print(np.mean([1 if np.where(row == i)[0] < n_clients-6 else -1 for row in maxcs]))
        #w[i] += np.mean([1 if np.where(row == i)[0] < n_clients-6 else 0 for row in maxcs])
    w /= n_clients
    
    vals = np.matmul(cs,w)
    
    wv = -1 * np.transpose(vals) + 6
    print(wv)
    
    #wv[wv > 1] = 1
    #wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    
    wv[(wv == 1)] = .99
    wv[(wv == 0)] = .01

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0
    
    #out = np.transpose(1- (vals - vals.min()) / (np.ptp(vals)))
    
    #out = np.ones((1,50))
    
    return wv
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, input):

        #out = fun(input)
        
        x = input.squeeze(0)
        x = x.permute(1, 0)
        w = fun(x)
        print(w)
        w = w / w.sum()
        out = torch.sum(x.permute(1, 0) * w, dim=1, keepdim=True)

        return out    
    
"""
class Net():
    def __init__(self):
        #self.path_to_net = "./aaa/attention.pt"
        super(Net, self).__init__()
    def main(self, deltas: list, model):
        stacked = utils.stackStateDicts(deltas)
        
        E, comp = fun(stacked)
        
        #param_trainable = utils.getTrainableParameters(model)
        #param_nontrainable = [param for param in stacked.keys() if param not in param_trainable]
        #for param in param_nontrainable:
        #    del stacked[param]
        #proj_vec = convert_pca._convertWithPCA(stacked)
        #print(proj_vec.shape)
        #model = MLP(proj_vec.shape[0] * 10, 10)
        #model.load_state_dict(torch.load(self.path_to_net))
        #model.eval()
        #x = proj_vec.unsqueeze(0)
        #beta = x.median(dim=-1, keepdims=True)[0]
        #weight = model.getWeight(beta, x)
        #weight = F.normalize(weight, p=1, dim=-1)
        #weight = weight[0, 0, :]
        #print(weight)
        
        weight = [0 if i< np.median(comp)/2.0 else 1 for i in comp]
        
        print(weight)
        
        #print(smp.cosine_similarity(E) - np.eye(50))
        
        Delta = deepcopy(deltas[0])
        param_float = utils.getFloatSubModules(Delta)

        for param in param_float:
            Delta[param] *= 0
            for i in range(len(deltas)):
                Delta[param] += deltas[i][param] * weight[i]
            Delta[param] = Delta[param]/len(deltas)    
        
        return Delta
        """