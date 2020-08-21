#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:17:53 2020

@author: SpaceMeerkat

This script takes the WISDOM moment maps and pushes them through the pretrained
network in training mode. This is a non-blind test script, meaning that the 
results may influence the network's ability to make predictions on test data
in the future unless performing all subsequent tests in non-blind mode as well.

"""

# =============================================================================
# Import relevant packages
# =============================================================================

import os 
import sys
sys.path.append("../training_files_2D/")
import torch
from tqdm import tqdm

from model2 import CAE
from WISDOM_utils import WISDOM_loader as load

# =============================================================================
# Setup the GPU training modes and device
# =============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.benchmark=True
torch.cuda.fastest =True
device = torch.device("cuda")

torch.cuda.empty_cache()

# =============================================================================
# Setup paths
# =============================================================================

model_path = '/home/corona/c1307135/Semantic_ML/Corellia/models/'
data_path = '/home/corona/c1307135/Semantic_ML/Corellia/WISDOM/fits_files/'

# =============================================================================
# Setup the model for retraining
# =============================================================================

model = CAE()
model.load_state_dict(torch.load(model_path+'semantic_AE_WISDOM_.pt'))

# =============================================================================
# Get the WISDOM tensors for trainins
# =============================================================================

mom0s, mom1s, pos = load(data_path = data_path)
mom0s = mom0s.to(device).to(torch.float)
mom1s = mom1s.to(device).to(torch.float)
pos = pos.to(device).to(torch.float)

# =============================================================================
# Setup the training parts
# =============================================================================

batch_size = mom0s.shape[0]
l = torch.arange(0 - 63/2., (63/2.)+1)
yyy, xxx, zzz = torch.meshgrid(l,l,l)
xxx, yyy, zzz = xxx.repeat(batch_size,1,1,1), yyy.repeat(batch_size,1,1,1), zzz.repeat(batch_size,1,1,1)
xxx = xxx.to(device).to(torch.float)
yyy = yyy.to(device).to(torch.float)
zzz = zzz.to(device).to(torch.float)

# =============================================================================
# Instantiate the model
# =============================================================================

model = CAE(xxx,yyy,zzz) 
model.to(device).to(torch.float)
model.train(True)

# =============================================================================
# Setup training parameters
# =============================================================================
    
loss_function=torch.nn.MSELoss()
learning_rate = 1e-6 # 0.975 // 2 for 400 epochs so (1e-4)*(0.975**200)
optim = torch.optim.Adam(model.parameters(),learning_rate)

# =============================================================================
# Setup the training run
# =============================================================================

save_model = True
    
epochs = 500

for epoch in tqdm(range(epochs)):
    prediction1, prediction2, inc, pos, vmax = model(mom0s,mom1s,pos)
    loss1 = loss_function(prediction1, mom0s)
    loss2 = loss_function(prediction2, mom1s)
    loss = loss1 + loss2
    optim.zero_grad(); loss.backward(); optim.step();
    
    if (save_model is not None) and (epoch == epochs-1):
            torch.save(model.state_dict(),model_path+'semantic_AE_non_blind_test_.pt')
            
# =============================================================================
# End of script
# =============================================================================








