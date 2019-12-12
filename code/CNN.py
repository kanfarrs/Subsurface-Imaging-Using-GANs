import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import utils

class CNN(nn.Module):

    def __init__(self, params, activation='tanh'):
        super(CNN, self).__init__()

        filters = params["filters"]    # list of filter sizes in_channel x out_channel x x_size x stride=X
        self.filters = filters
        self.conv = nn.ModuleList([])
        self.activation = []

    def forward(self, x, verbose=False):
        for i in range(len(self.conv)):
            x = self.conv[i](x)
            # if (verbose): print(x.size())
            # x = self.activation[i](x)
        return x

    def insertInstanceNorm(self,layers=[]):
        howmany = 0
        for layer in layers:
            where = layer + 1 + howmany
            self.conv.insert(where,nn.InstanceNorm2d(self.filters[layer-1][1]))
            howmany += 2

    def insertBatchNorm(self,layers=[]):
        howmany = 0
        for layer in layers:
            where = layer + 1 + howmany
            self.conv.insert(where,nn.BatchNorm2d(self.filters[layer-1][1]))
            howmany += 2

    def insertDropout(self,layers=[]):
        howmany = 0
        for layer in layers:
            where = layer + 1 + howmany
            self.conv.insert(where,nn.Dropout2d())
            howmany += 2

    def reset(self):
        for param in self.parameters():
            param.requires_grad = True

    def setNonTrainable(self):
        for param in self.parameters():
            param.requires_grad = False

    def showFilter(self,layer):
        filters = self.conv[layer].weight.detach().numpy()
        utils.showFilter(filters)

    def show_arch(self):
        for idx, m in enumerate(self.modules()):
            print(idx, '->', m)

    def count_params(self):
        total = sum(par.numel() for par in self.parameters())
        train = sum(par.numel() for par in self.parameters() if par.requires_grad)
        print("Total num of parameters: %d \nTrainable parameters: %d \nNon-Trainable parameters %d" %(total, train, total-train))

    def dryrun(self,input):
        o = input.shape[3]
        i=0
        for coef in self.filters:
            i += 1
            o = int((o + 2*int(coef[2]/2) - coef[2] - (coef[2]-1)*(coef[4]-1))/coef[3]) + 1
            print('Size after layer %s: %s' % (i,o))

    def receiptive(self,input):
        o = input.shape[3]
        i=0
        j=1
        r=1
        for coef in self.filters:
            i += 1
            o = int((o + 2*int(coef[2]/2) - coef[2] - (coef[2]-1)*(coef[4]-1))/coef[3]) + 1
            j = j * coef[3]
            r = r + (coef[2]-1)*j
            print('Receiptive field for layer %s: %s' % (i,r))

class Unet(CNN):
    def __init__(self, params, activation='tanh', padding=0):
        super().__init__(params,activation)
        for coef in self.filters:
            if len(coef) < 3: raise SystemExit('Filter is not correct size' % coef)
            CL = nn.Conv2d(coef[0],coef[1],coef[2],stride=coef[3],padding=int(coef[2]/2),dilation=coef[4])
            nn.init.xavier_uniform_(CL.weight)
            nn.init.zeros_(CL.bias)
            self.conv.append(CL)
            self.conv.append(nn.Tanh())
        for coef in reversed(self.filters):
            CL = nn.ConvTranspose2d(coef[1],coef[0],coef[2],stride=coef[3],output_padding=coef[3]-1,padding=int(coef[2]/2),dilation=coef[4])
            nn.init.xavier_uniform_(CL.weight)
            nn.init.zeros_(CL.bias)
            self.conv.append(CL)
            self.conv.append(nn.Tanh())

class Encoder(CNN):
    def __init__(self, params, activation='tanh'):
        super().__init__(params,activation)
        for coef in self.filters:
            if len(coef) < 3: raise SystemExit('Filter is not correct size' % coef)
            CL = nn.Conv2d(coef[0],coef[1],coef[2],stride=coef[3],padding=int(coef[2]/2),dilation=coef[4])
            nn.init.xavier_uniform_(CL.weight)
            nn.init.zeros_(CL.bias)
            self.conv.append(CL)
            self.conv.append(nn.LeakyReLU(negative_slope=0.2))
        # self.conv[len(self.conv)-1] = nn.Sigmoid()
        self.conv[len(self.conv)-1] = nn.ReLU()
