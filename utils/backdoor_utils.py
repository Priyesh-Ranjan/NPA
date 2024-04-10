from __future__ import print_function
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

'''
Modified upon
https://github.com/howardmumu/Attack-Resistant-Federated-Learning/blob/70db1edde5b4b9dfb75633ca5dd5a5a7303c1f4c/FedAvg/Update.py#L335


Reference:
Fu, Shuhao, et al. "Attack-Resistant Federated Learning with Residual-based Reweighting." arXiv preprint arXiv:1912.11464 (2019).

'''
import random
# def getRandomPattern(k=6,seed=0):
#     random.seed(seed)
#     c_range=[0,1,2]
#     xylim=6
#     x_range=list(range(xylim))
#     y_range=list(range(xylim))
#     x_offset=random.randint(0,32-xylim)
#     y_offset=random.randint(0,32-xylim)
#     x_range=list(map(lambda u: u+x_offset, x_range))
#     y_range=list(map(lambda u: u+y_offset, y_range))
    
#     combo = [c_range, x_range, y_range]
#     pattern = set()
#     while len(pattern) < k:
#         elem = tuple([random.choice(comp) for comp in combo])
#         pattern.add(elem)

#     return list(pattern)

def getRandomPattern(k=6,seed=2):
    #pattern=[[0, random.randint(0,6), random.randint(0,6)], [0, random.randint(0,6), random.randint(0,6)], [0, random.randint(0,6), random.randint(0,6)],\
    #         [0, random.randint(0,6), random.randint(0,6)], [0, random.randint(0,6), random.randint(0,6)], [0, random.randint(0,6), random.randint(0,6)],\
                                 
    #        [0, random.randint(0,6), random.randint(0,6)], [0, random.randint(0,6), random.randint(0,6)], [0, random.randint(0,6), random.randint(0,6)],\
    #        [0, random.randint(0,6), random.randint(0,6)], [0, random.randint(0,6), random.randint(0,6)], [0, random.randint(0,6), random.randint(0,6)], ]
    pattern=[[0, 0, 0], [0, 0, 1], [0, 0, 2],   [0, 0, 4], [0, 0, 5], [0, 0, 6],\
                                 
                                 [0, 2, 0], [0, 2, 1], [0, 2, 2],   [0, 2, 4], [0, 2, 5], [0, 2, 6], ]
    random.seed(2)
    #c=random.randint(0,3)
    c = 0
    xylim=6
    x_interval=random.randint(0,6)
    y_interval=random.randint(0,6)
    x_offset=random.randint(0,32-xylim-3)
    y_offset=random.randint(0,32-xylim-3)
    pattern=[[c,p[1]+x_offset,p[2]+y_offset] for p in pattern]
    pattern[3:6]=list(map(lambda p: [c,p[1],p[2]+y_interval],pattern[3:6]))
    pattern[-3:]=list(map(lambda p: [c,p[1],p[2]+y_interval],pattern[-3:]))    
    pattern[6:]=list(map(lambda p: [c,p[1]+x_interval,p[2]],pattern[6:]))
    return list(pattern)

def getDifferentPattern(y_offset, x_offset, y_interval=1, x_interval=1):
    pattern=[[0, 0, 0], [0, 0, 1], [0, 0, 2],   [0, 0, 4], [0, 0, 5], [0, 0, 6],\
                                 
                                 [0, 2, 0], [0, 2, 1], [0, 2, 2],   [0, 2, 4], [0, 2, 5], [0, 2, 6], ]
#     random.seed(seed)
#     c=random.randint(0,3)
    c=0
    xylim=6
    pattern=[[c,p[1]+x_offset,p[2]+y_offset] for p in pattern]
    pattern[3:6]=list(map(lambda p: [c,p[1],p[2]+y_interval],pattern[3:6]))
    pattern[-3:]=list(map(lambda p: [c,p[1],p[2]+y_interval],pattern[-3:]))    
    pattern[6:]=list(map(lambda p: [c,p[1]+x_interval,p[2]],pattern[6:]))      
    return list(pattern)
    
def getNonPersistantPattern(evaluation,part):
    
    pattern_1 = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 4, 0], [0, 4, 1], [0, 4, 2], [0, 4, 3], [0, 4, 4], 
                [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 1, 4], [0, 2, 4], [0, 3, 4], ] 
    
    #pattern_1 = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5], [0, 0, 6], [0, 0, 7], [0, 0, 8],  
    #            [0, 8, 0], [0, 8, 1], [0, 8, 2], [0, 8, 3], [0, 8, 4], [0, 8, 5], [0, 8, 6], [0, 8, 7], [0, 8, 8],
    #            [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0], [0, 5, 0], [0, 6, 0], [0, 7, 0],
    #            [0, 1, 8], [0, 2, 8], [0, 3, 8], [0, 4, 8], [0, 5, 8], [0, 6, 8], [0, 7, 8], ]
    
    pattern_2 = [[0, 2, 27], [0, 2, 28], [0, 2, 29], [0, 2, 30], [0, 2, 31], [0, 0, 29], [0, 1, 29], [0, 2, 29], [0, 3, 29], [0, 4, 29],]
    
    #pattern_2 = [[0, 4, 23], [0, 4, 24], [0, 4, 25], [0, 4, 26], [0, 4, 27], [0, 4, 28], [0, 4, 29], [0, 4, 30], [0, 4, 31], 
    #             [0, 0, 27], [0, 1, 27], [0, 2, 27], [0, 3, 27], [0, 4, 27], [0, 5, 27], [0, 6, 27], [0, 7, 27], [0, 8, 27], ]
    
    pattern_3 = [[0, 0, 13], [0, 0, 14], [0, 0, 15], [0, 0, 16], [0, 0, 17], [0, 2, 13], [0, 2, 14], [0, 2, 15], [0, 2, 16], [0, 2, 17], ]
                                 
    #pattern_3 = [[0, 0, 11], [0, 0, 12], [0, 0, 13], [0, 0, 14], [0, 0, 15], [0, 0, 16], [0, 0, 17], [0, 0, 18], [0, 0, 19],  
    #             [0, 4, 11], [0, 4, 12], [0, 4, 13], [0, 4, 14], [0, 4, 15], [0, 4, 16], [0, 4, 17], [0, 4, 18], [0, 4, 19], ]   
    
    #pattern_4 = [[0, 23, 13], [0, 24, 13], [0, 25, 13], [0, 26, 13], [0, 27, 13], [0, 28, 13], [0, 29, 13], [0, 30, 13], [0, 31, 13], 
    #             [0, 23, 17], [0, 24, 17], [0, 25, 17], [0, 26, 17], [0, 27, 17], [0, 28, 17], [0, 29, 17], [0, 30, 17], [0, 31, 17],]  
    
    pattern_4 = [[0, 27, 14], [0, 28, 14], [0, 29, 14], [0, 30, 14], [0, 31, 14], [0, 27, 16], [0, 28, 16], [0, 29, 16], [0, 30, 16], [0, 31, 16], ]
    
    pattern_5 = [[0, 27, 1], [0, 28, 1], [0, 29, 1], [0, 30, 1], [0, 31, 1], [0, 27, 3], [0, 28, 3], [0, 29, 3], [0, 30, 3], [0, 31, 3], 
                 [0, 28, 0], [0, 28, 1], [0, 28, 2], [0, 28, 3], [0, 28, 4], [0, 30, 0], [0, 30, 1], [0, 30, 2], [0, 30, 3], [0, 30, 4], ]
    
    #pattern_5 = [[0, 23, 2], [0, 24, 2], [0, 25, 2], [0, 26, 2], [0, 27, 2], [0, 28, 2], [0, 29, 2], [0, 30, 2], [0, 31, 2], 
    #             [0, 23, 5], [0, 24, 5], [0, 25, 5], [0, 26, 5], [0, 27, 5], [0, 28, 5], [0, 29, 5], [0, 30, 5], [0, 31, 5], 
    #             [0, 26, 0], [0, 26, 1], [0, 26, 2], [0, 26, 3], [0, 26, 4], [0, 26, 5], [0, 26, 6], [0, 26, 7], [0, 26, 8],
    #             [0, 29, 0], [0, 29, 1], [0, 29, 2], [0, 29, 3], [0, 29, 4], [0, 29, 5], [0, 29, 6], [0, 29, 7], [0, 29, 8], ]
    
    pattern_6 = [[0, 31, 31], [0, 30, 30], [0, 29, 29], [0, 28, 28], [0, 27, 27], [0, 29, 29], [0, 28, 30], [0, 27, 31], [0, 30, 28], [0, 31, 27], 
                 #[0, 29, 27], [0, 29, 28], [0, 29, 29], [0, 29, 30], [0, 29, 31], [0, 27, 29], [0, 28, 29], [0, 29, 29], [0, 30, 29], [0, 31, 29], 
                 ]
    
    #pattern_6 = [[0, 31, 31], [0, 30, 30], [0, 29, 29], [0, 28, 28], [0, 27, 27], [0, 26, 26], [0, 25, 25], [0, 24, 24], [0, 23, 23], 
    #             [0, 31, 23], [0, 30, 24], [0, 29, 25], [0, 28, 26], [0, 27, 27], [0, 26, 28], [0, 25, 29], [0, 24, 30], [0, 23, 31],]
    
    if evaluation :
        
        #pattern = pattern_1+pattern_2+pattern_3+pattern_4
        pattern=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 4, 0], [0, 4, 1], [0, 4, 2], [0, 4, 3], [0, 4, 4], 
                [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 1, 4], [0, 2, 4], [0, 3, 4],
                [0, 2, 27], [0, 2, 28], [0, 2, 29], [0, 2, 30], [0, 2, 31], [0, 0, 29], [0, 1, 29], [0, 2, 29], [0, 3, 29], [0, 4, 29], 
                [0, 0, 13], [0, 0, 14], [0, 0, 15], [0, 0, 16], [0, 0, 17], [0, 2, 13], [0, 2, 14], [0, 2, 15], [0, 2, 16], [0, 2, 17], 
                [0, 27, 14], [0, 28, 14], [0, 29, 14], [0, 30, 14], [0, 31, 14], [0, 27, 16], [0, 28, 16], [0, 29, 16], [0, 30, 16], [0, 31, 16], 
                [0, 27, 1], [0, 28, 1], [0, 29, 1], [0, 30, 1], [0, 31, 1], [0, 27, 3], [0, 28, 3], [0, 29, 3], [0, 30, 3], [0, 31, 3], 
                 [0, 28, 0], [0, 28, 1], [0, 28, 2], [0, 28, 3], [0, 28, 4], [0, 30, 0], [0, 30, 1], [0, 30, 2], [0, 30, 3], [0, 30, 4], 
                [0, 31, 31], [0, 30, 30], [0, 29, 29], [0, 28, 28], [0, 27, 27], [0, 29, 29], [0, 28, 30], [0, 27, 31], [0, 30, 28], [0, 31, 27], 
                 #[0, 29, 27], [0, 29, 28], [0, 29, 29], [0, 29, 30], [0, 29, 31], [0, 27, 29], [0, 28, 29], [0, 29, 29], [0, 30, 29], [0, 31, 29], 
                 ]
        #c=0
        #xylim=6
        #pattern=[[c,p[1],p[2]] for p in pattern]
        #pattern[3:6]=list(map(lambda p: [c,p[1],p[2]],pattern[3:6]))
        #pattern[-3:]=list(map(lambda p: [c,p[1],p[2]],pattern[-3:]))    
        #pattern[6:]=list(map(lambda p: [c,p[1],p[2]],pattern[6:]))
    else :
        #pattern = random.sample([pattern_1,pattern_2,pattern_3,pattern_4],1)[0]
        if part == 0 :
            pattern = pattern_1
        if part == 1 :
            pattern = pattern_2    
        if part == 2 :
            pattern = pattern_3
        if part == 3 :
            pattern = pattern_4   
        if part == 4 :
            pattern = pattern_5
        if part == 5 :
            pattern = pattern_6    
    return list(pattern)    

class Backdoor_Utils():

    def __init__(self):
        self.backdoor_label = 8
        #self.trigger_position = [[0, 0, 0], [0, 0, 1], [0, 0, 2],   [0, 0, 4], [0, 0, 5], [0, 0, 6],\
                                 
        #                         [0, 2, 0], [0, 2, 1], [0, 2, 2],   [0, 2, 4], [0, 2, 5], [0, 2, 6], ]
        #self.trigger_position = getRandomPattern(6,None)
        #self.trigger_position = getDifferentPattern(3,3)
        self.trigger_value = 1

    def get_poison_batch(self, data, targets, part, backdoor_fraction, backdoor_label, evaluation=False):
        #         poison_count = 0
        new_data = torch.empty(data.shape)
        new_targets = torch.empty(targets.shape)

        for index in range(0, len(data)):
            if evaluation:  # will poison all batch data when testing
                new_targets[index] = backdoor_label
                new_data[index] = self.add_backdoor_pixels(data[index], evaluation, part)
            #                 poison_count += 1
                #b = new_data[index][0].tolist()
                #if index == 0 :
                #        plt.imsave('backdoor/'+str(random.randint(0,50))+'.png', np.array(b).reshape(32,32), cmap=cm.gray)

            else:  # will poison only a fraction of data when training
                if torch.rand(1) < backdoor_fraction and part >= 0 :
                        new_targets[index] = backdoor_label
                        new_data[index] = self.add_backdoor_pixels(data[index], evaluation, part)
                #                     poison_count += 1
                    #print(new_data.shape)
                #    b = new_data[index][0].tolist()
                #    if index == 0 :
                #        plt.imsave('backdoor/'+str(random.randint(0,200))+'.png', np.array(b).reshape(32,32), cmap=cm.gray)
                        #print(self.trigger_position)
                    #    p = self.trigger_position
                
                
                
                else:
                    new_data[index] = data[index]
                    new_targets[index] = targets[index]

        new_targets = new_targets.long()
        if evaluation:
            new_data.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_data, new_targets

    def setRandomTrigger(self,k=6,seed=None):
        '''
        Use the default pattern if seed equals 0. Otherwise, generate a random pattern.
        '''
        if seed==0:
            return
        self.trigger_position=getRandomPattern(k,seed)

    def add_backdoor_pixels(self, item, evaluation, part):
        pos = getNonPersistantPattern(evaluation, part)
        for p in pos:
        #for i in range(0, 12):
            #pos = self.trigger_position[i]
                item[p[0]][p[1]][p[2]] = 1
        return item
    
    def setTrigger(self,x_offset,y_offset,x_interval,y_interval):
        self.trigger_position=getDifferentPattern(x_offset,y_offset,x_interval,y_interval)