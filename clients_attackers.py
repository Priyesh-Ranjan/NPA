from __future__ import print_function

import torch
import torch.nn.functional as F
from copy import deepcopy

from utils import utils
from utils.backdoor_semantic_utils import SemanticBackdoor_Utils
from utils.backdoor_utils import Backdoor_Utils
from clients import *       


class Attacker_Backdoor(Client):
    def __init__(self, cid, ctype, model, dataLoader, optimizer, criterion=F.nll_loss, device='cpu', inner_epochs=1, backdoor_scaling = 6, backdoor_fraction = 0.2):
        super(Attacker_Backdoor, self).__init__(cid, ctype, model, dataLoader, optimizer, criterion, device, inner_epochs, backdoor_scaling, backdoor_fraction)
        self.utils = Backdoor_Utils()

    def data_transform(self, data, target, epoch):
        data, target = self.utils.get_poison_batch(data, target, self.cid%6, backdoor_fraction=self.backdoor_fraction,
                                                   backdoor_label=self.utils.backdoor_label)
        return data, target
    
    def update(self, epoch):
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'
        newState = self.model.state_dict()
        for param in self.originalState:
            self.stateChange[param] = self.backdoor_scaling*(newState[param] - self.originalState[param])
        self.isTrained = False  
        
        
class Attacker_Distributed_Backdoor(Client):
    def __init__(self, cid, ctype, model, dataLoader, optimizer, criterion=F.nll_loss, device='cpu', inner_epochs=1, backdoor_scaling = 6, backdoor_fraction = 0.2):
        super(Attacker_Backdoor, self).__init__(cid, ctype, model, dataLoader, optimizer, criterion, device, inner_epochs, backdoor_scaling, backdoor_fraction)
        self.utils = Backdoor_Utils()

    def data_transform(self, data, target, epoch):
        if epoch%4 == self.cid/6 :
            data, target = self.utils.get_poison_batch(data, target, self.cid%6, backdoor_fraction=self.backdoor_fraction,
                                                   backdoor_label=self.utils.backdoor_label)
        return data, target
    
    def update(self, epoch):
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'
        newState = self.model.state_dict()
        if epoch%4 == self.cid/6 :
            for param in self.originalState:
                self.stateChange[param] = self.backdoor_scaling*(newState[param] - self.originalState[param])
        else :   
            for param in self.originalState:
                self.stateChange[param] = newState[param] - self.originalState[param]
        self.isTrained = False  