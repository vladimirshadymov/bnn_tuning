from __future__ import print_function
import csv
from itertools import zip_longest
import torch.nn as nn
from mnist import MnistDenseBNN
from binarized_layers import BinarizedLinear
from training_routines import tuning, train, test
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda-num', type=int, default=0,
                        help='Choses GPU number')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    

    torch.manual_seed(args.seed)

    device = torch.device("cuda:%d" % args.cuda_num if torch.cuda.is_available() else "cpu")
    print("Use device:", device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = MnistDenseBNN().to(device)

    test_accuracy = []
    train_accuracy = []

    # noise addition 
    model.set_noise_std(std=1)
    model.set_noise(True)

    model.load_state_dict(torch.load("../model/mnist_bnn.pt"))
    model.quantize_accumulative_weigths()

    for param in model.parameters():
        param.requires_grad = False

    params = []
    for layer in model.modules():
        if isinstance(layer, BinarizedLinear):
            layer.weight.requires_grad = True
            params.extend(layer.weight)
    
    # for param in model.parameters():
    #     if not param.requires_grad:
    #         param.detach_()
    

    # for p in model.parameters():
    #     p.requires_grad = False

    # for layer in model.modules():
    #     if isinstance(layer, BinarizedLinear):
    #         layer.weight.requires_grad = True

    # for p in model.parameters():
    #     print(p.requires_grad)

    optimizer = optim.SGD(params, lr=1.5)

    test(args, model, device, test_loader=test_loader, train_loader=train_loader)

    for epoch in range(1, args.epochs + 1):
        print('Epoch:', epoch)
        tuning(args, model, device, train_loader, optimizer, epoch, prob_rate=0)
        test(args, model, device, test_loader, train_loader, test_accuracy, train_accuracy)

    
if __name__ == '__main__':
    main()
