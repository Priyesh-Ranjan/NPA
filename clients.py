from __future__ import print_function

from copy import deepcopy

import torch
import torch.nn.functional as F
import random

class Client():
    def __init__(self, cid, ctype, model, dataLoader, optimizer, criterion=F.nll_loss, device='cpu', inner_epochs=1, backdoor_scaling=1, backdoor_fraction=0):
        self.cid = cid
        self.ctype = ctype
        self.model = model
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.device = device
        self.log_interval = len(dataLoader) - 1
        self.init_stateChange()
        self.originalState = deepcopy(model.state_dict())
        self.isTrained = False
        self.inner_epochs = inner_epochs
        self.criterion = criterion
        self.backdoor_scaling = backdoor_scaling
        self.backdoor_fraction = backdoor_fraction

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.stateChange = states

    def setModelParameter(self, states):
        self.model.load_state_dict(deepcopy(states))
        self.originalState = deepcopy(states)
        self.model.zero_grad()

    def data_transform(self, data, target, epoch):
        return data, target
    
    def backdoor_scaling(self):
        newState = self.model.state_dict()
        self.model.load_state_dict(deepcopy(newState)) 

    def train(self, epoch):
        self.model.to(self.device)
        self.model.train()
        for e in range(self.inner_epochs):
            for batch_idx, (data, target) in enumerate(self.dataLoader):
                data, target = self.data_transform(data, target, epoch)
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        
        self.isTrained = True
        self.model.cpu()  ## avoid occupying gpu when idle

    def test(self, testDataLoader):
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in testDataLoader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(testDataLoader.dataset)
        self.model.cpu()  ## avoid occupying gpu when idle
        print('client {} ## Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(self.cid,
                                                                                              test_loss, correct, len(
                testDataLoader.dataset),
                                                                                              100. * correct / len(
                                                                                                  testDataLoader.dataset)))
    
    def scaling(self, epoch) :
        newState = self.model.state_dict()
        for param in self.originalState:
            self.stateChange[param] = newState[param] - self.originalState[param]
            
    def update(self, epoch):
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'
        self.scaling(epoch)
        self.isTrained = False

    def getDelta(self):
        return self.stateChange
