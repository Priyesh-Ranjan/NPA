from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from clients_attackers import *
from server import Server

#schedule = {(0,1,2) : [2,3,4], (3,4,5) : [4,6,9], (6,7,8) : [2,5,9], (9,) : [8,9]}

def main(args):
    print('#####################')
    print('#####################')
    print('#####################')
    print(f'Aggregation Rule:\t{args.AR}\nData distribution:\t{args.loader_type}\nAttacks:\t{args.attacks} ')
    print('#####################')
    print('#####################')
    print('#####################')

    torch.manual_seed(args.seed)

    device = args.device

    attacks = args.attacks

    writer = SummaryWriter(f'./logs/{args.output_folder}/{args.experiment_name}')

    if args.dataset == 'mnist':
        from tasks import mnist
        trainData = mnist.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                           store=False)
        testData = mnist.test_dataloader(args.test_batch_size)
        Net = mnist.Net
        criterion = F.cross_entropy
    elif args.dataset == 'cifar':
        from tasks import cifar
        trainData = cifar.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                           store=False)
        testData = cifar.test_dataloader(args.test_batch_size)
        Net = cifar.Net
        criterion = F.cross_entropy
    elif args.dataset == 'fmnist':
        from tasks import fmnist
        trainData = fmnist.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                           store=False)
        testData = cifar.test_dataloader(args.test_batch_size)
        Net = fmnist.Net
        criterion = F.cross_entropy   

    # create server instance
    model0 = Net()
    server = Server(model0, testData, criterion, device, kappa=args.kappa, gamma=args.gamma, eps=args.eps, tau=args.tau)
    server.set_AR(args.AR)
    server.path_to_aggNet = args.path_to_aggNet
    if args.save_model_weights:
        server.isSaveChanges = True
        server.savePath = f'./AggData/{args.loader_type}/{args.dataset}/{args.attacks}/{args.AR}'
        from pathlib import Path
        Path(server.savePath).mkdir(parents=True, exist_ok=True)
        '''
        honest clients are labeled as 1, malicious clients are labeled as 0
        '''
        label = torch.ones(args.num_clients)
        for i in args.attacker_list:
            label[i] = 0

        torch.save(label, f'{server.savePath}/label.pt')
    # create clients instance

    attacker_list = args.attacker_list

    for i in range(args.num_clients):
        model = Net()
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        if i in attacker_list and "NPA" in args.attacks.upper():
            client_i = Attacker_NPA(i,'NP', model, trainData[i], optimizer, criterion, device, args.inner_epochs, args.backdoor_scaling, args.backdoor_fraction)
        elif i in attacker_list and 'NPA' not in args.attacks.upper():
            client_i = Attacker_PA(i,'P', model, trainData[i], optimizer, criterion, device, args.inner_epochs, args.backdoor_scaling, args.backdoor_fraction)
        else:
            client_i = Client(i,'N', model, trainData[i], optimizer, criterion, device, args.inner_epochs, 1, 0)
        server.attach(client_i)
    
    server.initialize()
    
    loss, accuracy = server.test()
    steps = 0
    writer.add_scalar('test/loss', loss, steps)
    writer.add_scalar('test/accuracy', accuracy, steps)

    if 'BACKDOOR' in args.attacks.upper():
        loss, accuracy = server.test_backdoor()

        writer.add_scalar('test/loss_backdoor', loss, steps)
        writer.add_scalar('test/backdoor_success_rate', accuracy, steps)

    for j in range(args.epochs):
        steps = j + 1

        print('\n\n########EPOCH %d ########' % j)
        print('###Model distribution###\n')
        server.distribute()
        #         group=Random().sample(range(5),1)
        group = range(args.num_clients)
        server.train(group, j)
        #         server.train_concurrent(group)

        loss, accuracy = server.test()

        writer.add_scalar('test/loss', loss, steps)
        writer.add_scalar('test/accuracy', accuracy, steps)

        if 'BACKDOOR' in args.attacks.upper():
            loss, accuracy = server.test_backdoor()

            writer.add_scalar('test/loss_backdoor', loss, steps)
            writer.add_scalar('test/backdoor_success_rate', accuracy, steps)

    writer.close()
