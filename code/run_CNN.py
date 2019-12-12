import sys
import numpy as np
sys.path.append('../code/')
sys.path.append('../cGAN/Code/')

import utilities as ut
data = np.load('../Patch/train_set_marm_100-30.npy')
focused, defocused = ut.dataProcess(data)

import CNN

param = {
    "filters" : [[1,16,11,1,1],
                 [16,32,7,2,1],
                 [32,64,5,1,1],
                 [64,64,3,1,1],
                 [64,64,5,2,1],
                 [64,64,3,1,1],
                 [64,64,3,1,1],
                ]
}
net = CNN.Unet(param)

import torch
import numpy as np
import torch.nn as nn
import Optimization as Opt
import Problem as Prob

f = torch.from_numpy(np.transpose(focused,(0,3,1,2))).float()
d = torch.from_numpy(np.transpose(defocused,(0,3,1,2))).float()
# target = torch.randn(output.shape)  # a dummy target, for example
probL2 = Prob.Problem(d,f,net,valid_perc=0.1,gpu=True,gpu_id=3)

# batch_size = 200
batch_size = 5
num_iter = 201
# lr = 0.005
lr = 0.001
eps = 1e-4
solver = Opt.Solver(probL2,batch_size=batch_size,learning_rate=lr,
                    max_iter=num_iter,flush_res=100,flush_file='cnn.pt',
                    eps=eps)
solver.run()
