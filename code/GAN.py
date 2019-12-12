import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import CNN
import utils

class Chain(nn.Module):

    def __init__(self, net1, net2):
        super(Chain, self).__init__()

        self.net1 = net1
        self.net2 = net2

    def forward(self, x, verbose=False):
        x = self.net1.forward(x,verbose)
        x = self.net2.forward(x,verbose)
        return x

    def reset(self):
        for param in self.parameters():
            param.requires_grad = True

    def train(self,net):
        if net==1: self.net2.setNonTrainable()
        if net==2: self.net1.setNonTrainable()

class GAN(nn.Module):

    def __init__(self, params_G, params_D):
        super(GAN, self).__init__()

        self.fake = None
        self.G = CNN.Unet(params_G)
        self.D = CNN.Encoder(params_D)
        # add changing the last activation in D or maybe all?

    def forward(self, x, verbose=False):
        self.fake = self.G.forward(x,verbose)
        x = self.D.forward(self.fake,verbose)
        return x

    def getFake(self):
        return self.fake

    def reset(self):
        for param in self.parameters():
            param.requires_grad = True

    def train(self,net):
        self.reset()
        if (net=='G'):
            self.D.setNonTrainable()
        if (net=='D'):
            self.G.setNonTrainable()


    def show_arch(self):
        for idx, m in enumerate(self.modules()):
            print(idx, '->', m)

    def count_params(self):
        total = sum(par.numel() for par in self.parameters())
        train = sum(par.numel() for par in self.parameters() if par.requires_grad)
        print("Total num of parameters: %d \nTrainable parameters: %d \nNon-Trainable parameters %d" %(total, train, total-train))


class cycleGAN(nn.Module):

    def __init__(self, params_G, params_D):
        super(cycleGAN, self).__init__()

        self.GAN = nn.ModuleList([])
        self.Cycle = nn.ModuleList([])

        self.GAN.append(GAN(params_G,params_D))
        self.GAN.append(GAN(params_G,params_D))

        self.Cycle.append(Chain(self.getGAN(1).G, self.getGAN(2).G))
        self.Cycle.append(Chain(self.getGAN(2).G, self.getGAN(1).G))

    def getGAN(self,which):
        return self.GAN[which-1]

    def getCycle(self,which):
        return self.Cycle[which-1]

    def show_arch(self):
        for idx, m in enumerate(self.modules()):
            print(idx, '->', m)

    def count_params(self):
        total = sum(par.numel() for par in self.parameters())
        train = sum(par.numel() for par in self.parameters() if par.requires_grad)
        print("Total num of parameters: %d \nTrainable parameters: %d \nNon-Trainable parameters %d" %(total, train, total-train))
