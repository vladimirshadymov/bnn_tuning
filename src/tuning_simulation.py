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

    parser.add_argument('--model-size', type=int, default=100, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    HIDDEN_SIZE = args.model_size 

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

    model = MnistDenseBNN(HIDDEN_SIZE).to(device)

    test_accuracy = []
    train_accuracy = []


    model.load_state_dict(torch.load(f"../model/mnist_bnn_{HIDDEN_SIZE}.pt"))
    model.quantize_accumulative_weigths()

    # noise addition 
    model.set_noise_std(std=0.5)
    model.set_noise(True)
    model.add_bit_error(bit_error_rate = 1e-1)

    for param in model.parameters():
        param.requires_grad_(False)

    params = []
    for layer in model.modules():
        if isinstance(layer, BinarizedLinear):
            layer.weight.requires_grad_(True)
            params.append(layer.weight)


    optimizer = optim.Adam(params, lr=1.5)

    test(args, model, device, test_loader=test_loader, train_loader=train_loader)

    prob_rate_arr = np.flip(np.arange(-7, -3.9, 0.5))

    d_train = []
    d_test = []

    print(f"Model hidden layer size: {HIDDEN_SIZE}")

    for p in prob_rate_arr:
        model.load_state_dict(torch.load(f"../model/mnist_bnn_{HIDDEN_SIZE}.pt"))
        model.quantize_accumulative_weigths()

        # noise addition 
        model.set_noise_std(std=0.5)
        model.set_noise(True)
        model.add_bit_error(bit_error_rate = 10**p)

        test_accuracy = [10**p]
        train_accuracy = [10**p]

        print(f'Probability rate: {p}')

        test(args, model, device, test_loader, train_loader, test_accuracy, train_accuracy)

        for epoch in range(1, np.int(10**(prob_rate_arr[0] - p))*args.epochs + 1):
            print('Epoch:', epoch)
            tuning(args, model, device, train_loader, optimizer, epoch, prob_rate=p)
            test(args, model, device, test_loader, train_loader, test_accuracy, train_accuracy)

        d_test.append(test_accuracy)
        d_train.append(train_accuracy)
        
        export_data = zip_longest(*d_train, fillvalue='')
        with open(f'../log/mnist_bnn_tuning_train_{HIDDEN_SIZE}.csv', 'w', encoding="ISO-8859-1", newline='') as report_file:
            wr = csv.writer(report_file)
            wr.writerows(export_data)
        report_file.close()

        export_data = zip_longest(*d_test, fillvalue='')
        with open(f'../log/mnist_bnn_tuning_test_{HIDDEN_SIZE}.csv', 'w', encoding="ISO-8859-1", newline='') as report_file:
            wr = csv.writer(report_file)
            wr.writerows(export_data)
        report_file.close()

if __name__ == '__main__':
    main()
