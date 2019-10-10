#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:00:38 2019

@author: SpaceMeerkat
"""

#_____________________________________________________________________________#
#_____________________________________________________________________________#

import os
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from cube_generator import cube_generator
from model import CovNet
from functions import learning_rate, return_cube

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.benchmark=True
torch.cuda.fastest =True

np.random.seed(0)

#_____________________________________________________________________________#
#_____________________________________________________________________________#
    
def training(model:torch.nn.Module,batch_size,epochs,loss_function,initial_lr):
    
    #_________________________________________________________________________#
    #~~~ SET UP THE MODEL   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #_________________________________________________________________________#
    
    device = torch.device("cuda")
    model = model.to(device).to(torch.float)
    
    for epoch in range(epochs):
        
        print("Epoch {0} of {1}" .format( (epoch+1), epochs))
        optim = torch.optim.Adam(model.parameters(),learning_rate(initial_lr,epoch))

        #_________________________________________________________________________#
        #~~~ CREATE THE BATCH  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #_________________________________________________________________________#
        
        index = np.arange(0,batch_size,1)
        
        num_cores = 14
        pool = mp.Pool(num_cores)
        results = list(tqdm(pool.imap(return_cube,index),total=len(index)))
        results = np.array(results)
        batch = np.array(results[:,0])
#        targets = results[:,1][0]
#        
        batch = torch.tensor(batch)
#        targets = torch.tensor(results[:,1])
        print(batch.shape)
#        print(targets.shape)
#        
#        batch = batch.to(device).to(torch.float)
#        batch.view(batch_size,batch.shape[-1],batch.shape[1],batch.shape[2])
        
        #_________________________________________________________________________#
        #~~~ TRAIN THE MODEL   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #_________________________________________________________________________#
        
        #prediction = model(batch,targets)

#_____________________________________________________________________________#
#_____________________________________________________________________________#
    
model = CovNet(55)
training(model,batch_size=5,epochs=1,loss_function=torch.nn.MSELoss(),initial_lr=1e-3)






