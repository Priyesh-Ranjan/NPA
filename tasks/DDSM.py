from __future__ import print_function

import pickle

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from dataloader import *
import os
from torchvision.io import read_image



class Net(nn.Module):
    #num_classes = 10
    #model = resnet18(pretrained=True)
    #n = model.fc.in_features
    #model.fc = nn.Linear(n, num_classes)
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,   64,  3)
        self.conv2 = nn.Conv2d(64,  128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 512)
        self.drop3 = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 24 * 24)
        #x = self.drop1(x)
        x = F.relu(self.fc1(x))
        #x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.drop3(x)
        x = F.sigmoid(self.fc3(x))
        return x


class DDSMDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.targets = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.targets.iloc[idx, 0])
        image = read_image(img_path)
        image = torch.as_tensor(np.asarray(image), dtype=torch.float32) 
        label = self.targets.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def getDataset():
    dataset = DDSMDataset('./data/DDSM/Train_Annotations/Annotations.csv', './data/DDSM/Train',
                            transform=transforms.Compose([transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]))
    #dataset = datasets.CIFAR10('./data',train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]))
    #dataset = DataLoader(datasets, batch_size=4, shuffle=True)
    return dataset


def basic_loader(num_clients, loader_type):
    dataset = getDataset()
    return loader_type(num_clients, dataset)


def train_dataloader(num_clients, loader_type='iid', store=True, path='./data/loader.pk'):
    assert loader_type in ['iid', 'byLabel',
                           'dirichlet'], 'Loader has to be one of the  \'iid\',\'byLabel\',\'dirichlet\''
    if loader_type == 'iid':
        loader_type = iidLoader
    elif loader_type == 'byLabel':
        loader_type = byLabelLoader
    elif loader_type == 'dirichlet':
        loader_type = dirichletLoader

    if store:
        try:
            with open(path, 'rb') as handle:
                loader = pickle.load(handle)
        except:
            print('loader not found, initialize one')
            loader = basic_loader(num_clients, loader_type)
    else:
        print('initialize a data loader')
        loader = basic_loader(num_clients, loader_type)
    if store:
        with open(path, 'wb') as handle:
            pickle.dump(loader, handle)

    return loader


def test_dataloader(test_batch_size):
    #test_loader = torch.utils.data.DataLoader(
    #    datasets.CIFAR10('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),
    #                                                                          transforms.Normalize((0.5,0.5,0.5),
    #                                                                                               (0.5,0.5,0.5))])),
    #    batch_size=test_batch_size, shuffle=True)
    test_loader = DataLoader(DDSMDataset('./data/DDSM/Test_Annotations/Annotations.csv', './data/DDSM/Test',
                            transform=transforms.Compose([transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])), 
                            batch_size=test_batch_size, shuffle=True)
    
    return test_loader


if __name__ == '__main__':
    from torchsummary import summary

    print("#Initialize a network")
    net = Net()
    summary(net.cuda(), (3, 50, 50))

    print("\n#Initialize dataloaders")
    loader_types = ['iid', 'dirichlet']
    for i in range(len(loader_types)):
        loader = train_dataloader(50, loader_types[i], store=False)
        print(f"Initialized {len(loader)} loaders (type: {loader_types[i]}), each with batch size {loader.bsz}.\
        \nThe size of dataset in each loader are:")
        print([len(loader[i].dataset) for i in range(len(loader))])
        print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")

    print("\n#Feeding data to network")
    x = next(iter(loader[i]))[0].cuda()
    y = net(x)
    print(f"Size of input:  {x.shape} \nSize of output: {y.shape}")
