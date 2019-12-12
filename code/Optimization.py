import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import utils

class Solver():

    def __init__(self,problem,batch_size=0,method=None,learning_rate=1e-5, momentum=0.9,
                max_iter=1,flush_res=1,flush_file=None,
                lr_each_iter=[100],lr_decay=1,eps=0):

        self.problem = problem
        self.num_batches = int(self.problem.input.shape[0]/batch_size)
        self.batch_size = int(batch_size)
        if (method=='sgd'):
            self.optimizer = optim.SGD(problem.net.parameters(), lr=learning_rate, momentum=momentum,nesterov=True)
        else:
            self.optimizer = optim.Adam(problem.net.parameters(), lr=learning_rate, weight_decay=eps)

        # self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, lr_each_iter, gamma=lr_decay, last_epoch=-1)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_each_iter , gamma=lr_decay, last_epoch=-1)
        # if method is not None: self.optimizer =
        self.max_iter = max_iter
        self.flush_res = flush_res
        self.flush_file = flush_file
        self.loss = np.zeros((2,self.max_iter))


    def run(self):
        self.problem.net.train()
        for it in range(self.max_iter):
            ind1 = torch.randperm(self.problem.input.shape[0])
            self.problem.input[:] = self.problem.input[ind1,:,:,:]
            # np.random.shuffle(self.problem.input)
            for ib in range(self.num_batches):
                utils.printProgressBar(ib,self.num_batches)

                start = ib*self.batch_size
                end = (ib+1)*self.batch_size
                # batchIn = np.expand_dims(self.problem.input[start:end,0,:,:],axis=1)
                # batchTarget = np.expand_dims(self.problem.input[start:end,1,:,:],axis=1)
                batchIn = self.problem.input[start:end,0,:,:].unsqueeze(1)
                batchOut = self.problem.input[start:end,1,:,:].unsqueeze(1)

                self.optimizer.zero_grad()
                out = self.problem.forward(batchIn)
                loss = self.problem.loss(out,batchOut)
                self.loss[0,it] += loss.item() / self.num_batches
                loss.backward()
                self.optimizer.step()

            loss = self.problem.loss_valid()
            self.loss[1,it] = loss.item()

            print("Iteration %s: train_loss = %f, valid_loss = %f" % (it, self.loss[0,it], self.loss[1,it]))
            if(it % self.flush_res==0):
                torch.save(self.problem.net, self.flush_file + '%s' % (it))
                np.save('cnn_loss.npy',self.loss)
            self.lr_scheduler.step()


class Solver_cycleGAN():

    def __init__(self,problem,batch_size=0,method=None,learning_rate=1e-5, momentum=0.9,
                max_iter=1,flush_res=1,flush_file=None,
                lr_each_iter=[100],lr_decay=1,eps=0, wasserstein=False, clip=0.01, sub_iter=1,
                loss_weight = {'D':1, 'G':1, 'C':1}):

        self.model_save = {}
        self.model = problem.model
        self.device = problem.device
        self.problem = problem.term
        self.optimizer = {}
        self.loss = {}
        self.num_batches = int(self.problem['D1_fake'].input.shape[0]/batch_size)
        self.batch_size = int(batch_size)

        self.max_iter = max_iter
        self.flush_res = flush_res
        self.flush_file = flush_file
        self.loss_weight = loss_weight

        self.sub_iter = sub_iter

        for key,prob in self.problem.items():
            lr = learning_rate
            if '2' in key: lr = 10*learning_rate
            # self.optimizer[key] = optim.Adam(prob.net.parameters(), lr=lr, weight_decay=eps)
            self.optimizer[key] = optim.RMSprop(prob.net.parameters(), lr=lr, weight_decay=eps, momentum=momentum)
            self.loss[key] = np.zeros(self.max_iter)

        # self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, lr_each_iter, gamma=lr_decay, last_epoch=-1)
        # self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_each_iter , gamma=lr_decay, last_epoch=-1)
        # if method is not None: self.optimizer =

    def run(self):

        for it in range(self.max_iter):
            t0 = time.time()

            ind1 = torch.randperm(self.problem['Cycle_1'].input.shape[0])
            ind2 = torch.randperm(self.problem['Cycle_2'].input.shape[0])
            self.problem['Cycle_1'].input[:] = self.problem['Cycle_1'].input[ind1,:,:,:]
            self.problem['Cycle_2'].input[:] = self.problem['Cycle_2'].input[ind2,:,:,:]

            for ib in range(self.num_batches):
                utils.printProgressBar(ib,self.num_batches)

                start = ib*self.batch_size
                end = (ib+1)*self.batch_size

                for prob in self.problem:

                    batchIn = self.problem[prob].input[start:end,0,:,:].unsqueeze(1)
                    batchOut = self.problem[prob].target[start:end,0,:,:].unsqueeze(1)

                    self.optimizer[prob].zero_grad()
                    self.loss[prob][it] += self.update(prob,batchIn,batchOut) / self.num_batches

            print("Time elapsed %f s" % (time.time()-t0))
            print("""
                    Iteration %s: D1_real = %f, D1_fake = %f, G1 = %f, \n\t
                                  D2_real = %f, D2_fake = %f, G2 = %f, \n\t
                                  Cycle_1 = %f, Cycle_2 = %f \n
                  """ %
                  (it, self.loss['D1_real'][it], self.loss['D1_fake'][it],self.loss['G1'][it],
                       self.loss['D2_real'][it], self.loss['D2_fake'][it],self.loss['G2'][it],
                       self.loss['Cycle_1'][it], self.loss['Cycle_2'][it])
                  )

            if(it % self.flush_res==0):
                mod = self.model.to('cpu')
                torch.save(mod, self.flush_file + '%s' %it)
                np.save('loss.npy',self.loss)
                self.model.to(self.device)
            # self.lr_scheduler.step()

    def update(self,name,batchIn,batchOut):
        if "real" in name:
            self.problem[name].net.reset()
            l = self.basic_update(name,batchIn,batchOut,self.loss_weight['D'])

        if "fake" in name:
            self.problem[name].net.train('D')
            l = self.basic_update(name,batchIn,batchOut,self.loss_weight['D'])
            self.problem[name].net.reset()

        if "G" in name:
            self.problem[name].net.train('G')
            for i in range(self.sub_iter):
                l = self.basic_update(name,batchIn,batchOut,self.loss_weight['G'])
            self.problem[name].net.reset()

        if "Cycle" in name:
            l = self.cycle_update(name,batchOut,self.loss_weight['C'])
            self.problem[name].net.reset()

        if "Id" in name:
            self.problem[name].net.reset()
            l = self.basic_update(name,batchIn,batchOut,self.loss_weight['G'])

        return l

    def basic_update(self,name,batchIn,batchOut,w):
        out = self.problem[name].forward(batchIn)
        l = self.problem[name].loss(out,batchOut)
        loss = w * l
        loss.backward(retain_graph=True)
        self.optimizer[name].step()

        return l.item()

    def cycle_update(self,name,batchOut,w):
        if '1' in name:
            fake = self.problem['G1'].net.getFake()
        if '2' in name:
            fake = self.problem['G2'].net.getFake()
        out = self.problem[name].net.net2.forward(fake)
        l = self.problem[name].loss(out,batchOut)
        loss = w * l
        self.problem[name].net.train(2)
        loss.backward(retain_graph=True)
        self.problem[name].net.train(1)
        loss.backward(retain_graph=True)
        self.optimizer[name].step()

        return l.item()
