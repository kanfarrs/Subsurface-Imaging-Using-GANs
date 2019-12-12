import sys
import numpy as np
sys.path.append('../code/')
sys.path.append('../cGAN/Code/')

import utilities as ut
data = np.load('../Patch/train_set_marm_100-30.npy')
focused, defocused = ut.dataProcess(data)

import CNN
import GAN
param_G = {
    "filters" : [[1,16,11,1,1],
                 [16,32,7,2,1],
                 [32,64,5,1,1],
                 [64,64,3,1,1],
                 [64,64,5,2,1],
                 [64,64,3,1,1],
                 [64,64,3,1,1],
                ]
}
# param_D = {
#     "filters" : [[1,16,11,1,1],
#                  [16,32,7,2,2],
#                  [32,64,5,1,3],
#                  [64,1,3,1,4],
#                 ]
# }

param_D = {
    "filters" : [[1,16,11,1,1],
                 [16,32,7,2,2],
                 [32,64,5,2,2],
                 [64,128,3,1,3],
                 [128,1,3,2,3],
                ]
}

cycleGan = GAN.cycleGAN(param_G,param_D)
cycleGan.getGAN(1).D.insertDropout([1,2,3])
cycleGan.getGAN(2).D.insertDropout([1,2,3])

import torch
import numpy as np
import torch.nn as nn
import Problem as Prob
import Optimization as Opt

f = torch.from_numpy(np.transpose(focused,(0,3,1,2))).float()
d = torch.from_numpy(np.transpose(defocused,(0,3,1,2))).float()

cycleGanProb = Prob.Problem_cycleGAN(d,f,cycleGan,gpu=True,gpu_id=1)

batch_size = 5
lr = 1e-5
num_iter = 1001
eps = 1e-8

solver = Opt.Solver_cycleGAN(cycleGanProb,
                    batch_size=batch_size,learning_rate=lr,max_iter=num_iter,flush_res=10,flush_file='model.pt'
                   ,eps=eps,loss_weight={'D':1e-3, 'G':1, 'C':10},sub_iter=1,momentum=0.1)
solver.run()
