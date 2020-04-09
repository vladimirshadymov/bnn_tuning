# Import PyTorch
import torch  # import main library
from torch.autograd import Function  # import Function to create custom activations
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

class BinarizeFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # save input for backward pass

        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = input.clone()
        grad_input[torch.abs(input) <= 1.] = 1.
        grad_input[torch.abs(input) > 1.] = 0.
        grad_input = grad_input * grad_output

        return grad_input

class Binarization(nn.Module):

    def __init__(self, min=-1, max=1, stochastic=False):
        super(Binarization, self).__init__()
        self.stochastic = stochastic
        self.min = min
        self.max = max

    def forward(self, input):
        return 0.5*(BinarizeFunction.apply(input)*(self.max - self.min) + self.min + self.max)


class BinarizedLinear(nn.Linear):

    def __init__(self, min_weight=-1, max_weight=1, *kargs, **kwargs):
        super(BinarizedLinear, self).__init__(*kargs, **kwargs)
        self.binarization = Binarization(min=min_weight, max=max_weight)
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.noise_on = False
        self.noise_std = 0.2
        self.noise = torch.normal(mean=0.0, std=torch.ones_like(self.weight.data)*self.noise_std)

    def forward(self, input):
        device_num = self.weight.get_device()
        device = torch.device("cuda:%d" % device_num)
        self.noise = self.noise.to(device)
        self.weight.data = nn.functional.hardtanh_(self.weight.data) 
        
        if self.noise_on:
            out = nn.functional.linear(input, self.binarization(self.weight)+self.noise, bias=self.bias)
        else:
            out = nn.functional.linear(input, self.binarization(self.weight), bias=self.bias)  # linear layer with binarized weights
        return out

    def quantize_accumulative_weigths(self):
        self.weight.data = self.binarization(self.weight.data)
        return
    
    def set_noise_std(self, std=0.2):
        self.noise_std = std
        self.noise = torch.normal(mean=0.0, std=torch.ones_like(self.weight.data)*self.noise_std)
        return
    
    def set_noise(self, noise_on=True):
        self.noise_on = noise_on
        return
    
    def calc_prop_grad(self, prob_rate=0):
        with torch.no_grad():
            tmp = torch.abs(self.weight.grad.data).add_(1e-10).clone()
            self.weight.grad.data.div_(tmp) # norm of grad values
            tmp = F.tanh(prob_rate*tmp).clone()
            tmp = torch.bernoulli(tmp).clone()
            self.weight.grad.data.mul_(tmp)
            # self.weight.grad.mul_(0)
            del tmp
            # print(self.weight)
        return

    def add_bit_error(self, bit_error_rate = 0):
        probs = torch.ones_like(self.weight.data).mul_(1 - bit_error_rate) # switching probabilities
        switching_tensor = torch.bernoulli(probs).mul(2.).add(-1.)
        self.weight.data.mul_(switching_tensor)
        return

