from __future__ import print_function
import csv
from itertools import zip_longest
import torch.nn as nn
from binarized_layers import BinarizedLinear, Binarization
from training_routines import train, test
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np

class MnistDenseBNN(nn.Module):
    def __init__(self, hidden_layer_size=100):
        super(MnistDenseBNN, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.layer1 = nn.Sequential(
            BinarizedLinear(in_features=28*28, out_features=HIDDEN_SIZE),
            nn.Dropout(0.2),
            nn.BatchNorm1d(HIDDEN_SIZE),
            Binarization()
        )

        self.layer2 = nn.Sequential(
            BinarizedLinear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            nn.Dropout(0.2),
            nn.BatchNorm1d(HIDDEN_SIZE),
            Binarization()
        )

        self.layer3 = nn.Sequential(
            BinarizedLinear(in_features=HIDDEN_SIZE, out_features=10),
            nn.Dropout(0.2),
            nn.BatchNorm1d(10)
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def quantize_accumulative_weigths(self):
        for layer in self.modules():
            if isinstance(layer, BinarizedLinear):
                layer.quantize_accumulative_weigths()
    
    def set_noise_std(self, std=0.2):
        for layer in self.modules():
            if isinstance(layer, BinarizedLinear):
                layer.set_noise_std(std=std)
    
    def set_noise(self, noise_on=True):
        for layer in self.modules():
            if isinstance(layer, BinarizedLinear):
                layer.set_noise(noise_on=noise_on)
    
    def calc_prop_grad(self, prob_rate=0):
        for layer in self.modules():
            if isinstance(layer, BinarizedLinear):
                layer.calc_prop_grad(prob_rate=prob_rate)
    
    def add_bit_error(self, bit_error_rate = 0):
        for layer in self.modules():
            if isinstance(layer, BinarizedLinear):
                layer.add_bit_error(bit_error_rate = bit_error_rate)

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
    
    HIDDEN_SIZE = 100

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
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    test_accuracy = []
    train_accuracy = []

    for epoch in range(1, args.epochs + 1):
        print('Epoch:', epoch, 'LR:', scheduler.get_lr())
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, train_loader, test_accuracy, train_accuracy)
        scheduler.step(epoch=epoch)
        
        if epoch>=10:
            if (args.save_model):
                torch.save(model.state_dict(), f"../model/mnist_bnn_{HIDDEN_SIZE}.pt")
            d = [train_accuracy, test_accuracy]
            export_data = zip_longest(*d, fillvalue='')
            with open(f'../model/mnist_bnn_report_{HIDDEN_SIZE}.csv', 'w', encoding="ISO-8859-1", newline='') as report_file:
                wr = csv.writer(report_file)
                wr.writerow(("Train accuracy", "Test accuracy"))
                wr.writerows(export_data)
            report_file.close()

if __name__ == '__main__':
    main()
