import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


class Problem():
    def __init__(self,input,target,net,valid_perc=0,
                 valid_set_name='valid_set.pt',train_set_name='train_set.pt',
                 gpu=False,gpu_id=1):

        # device = torch.device("cuda:%s" % (gpu_id) if (gpu and torch.cuda.is_available()) else "cpu")
        # if (not 'cuda' in net.device):
        self.net = net

        if (valid_perc > 0):
            num_valid = int(valid_perc * input.shape[0])
            indexes = np.random.choice(input.shape[0],input.shape[0],replace=False)
            self.valid_set = torch.from_numpy(np.concatenate((input[indexes[0:num_valid]],target[indexes[0:num_valid]]),axis=1))
            self.valid_set = self.valid_set.to(device)
            self.input = torch.from_numpy(np.concatenate((input[indexes[num_valid:]],target[indexes[num_valid:]]),axis=1))
            self.input = self.input.to(device)
            self.target = target
            self.target = self.target.to(device)
        else:
            self.input = input
            self.target = target
        # dummy = net(input)
        # if dummy.shape != target.shape: raise SystemExit('The CNN output shape is different from the target shape!')
        # self.loss = nn.MSELoss()


        if (valid_perc > 0): torch.save(self.valid_set,valid_set_name)
        torch.save(self.input,train_set_name)

    def forward(self,x):
        return self.net(x)

    def predict_train(self):
        return self.net(self.input[:,0,:,:].unsqueeze(1))

    def predict_valid(self):
        return self.net(self.valid_set[:,0,:,:].unsqueeze(1))

    def loss_valid(self):
        valid_pred = self.predict_valid()
        return self.loss(valid_pred,self.valid_set[:,1,:,:].unsqueeze(1))

class L2Problem(Problem):
    def __init__(self,input,target,net):
        super().__init__(input,target,net)
        self.loss = nn.MSELoss()


class L1Problem(Problem):
    def __init__(self,input,target,net):
        super().__init__(input,target,net)
        self.loss = nn.L1Loss()

class Problem_cycleGAN():

    def __init__(self, input1, input2, cycleGAN, gpu=False,gpu_id=0):

        device = torch.device("cuda:%s" % (gpu_id) if (gpu and torch.cuda.is_available()) else "cpu")

        self.term = {}
        self.model = cycleGAN.to(device)
        self.device = device

        input1 = input1.to(device)
        input2 = input2.to(device)

        out = self.model.getGAN(1).forward(input1)

        labels_real = torch.zeros(out.shape)
        labels_real_soft = torch.zeros(out.shape) + .1
        labels_fake = torch.ones(out.shape)

        labels_fake = labels_fake.to(device)
        labels_real = labels_real.to(device)
        labels_real_soft = labels_real_soft.to(device)

        self.term['D1_real'] = L2Problem(input2,labels_real,cycleGAN.getGAN(1).D)
        self.term['D1_fake'] = L2Problem(input1,labels_fake,cycleGAN.getGAN(1))
        self.term['G1'] = L2Problem(input1,labels_real,cycleGAN.getGAN(1))

        self.term['D2_real'] = L2Problem(input1,labels_real,cycleGAN.getGAN(2).D)
        self.term['D2_fake'] = L2Problem(input2,labels_fake,cycleGAN.getGAN(2))
        self.term['G2'] = L2Problem(input2,labels_real,cycleGAN.getGAN(2))

        self.term['Cycle_1'] = L1Problem(input1,input1,cycleGAN.getCycle(1))
        self.term['Cycle_2'] = L1Problem(input2,input2,cycleGAN.getCycle(2))

        self.term['Id_1'] = L1Problem(input2,input2,cycleGAN.getGAN(1).G)
        self.term['Id_2'] = L1Problem(input1,input1,cycleGAN.getGAN(2).G)
