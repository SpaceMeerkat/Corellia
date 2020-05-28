#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:00:38 2019

@author: SpaceMeerkat
"""

#_____________________________________________________________________________#
#_____________________________________________________________________________#

import multiprocessing as mp
import numpy as np
import os
from tqdm import tqdm
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Bfunctions import learning_rate, return_cube, plotter
from Bmodel import CAE
from cube_generator import cube_generator
from sauron_colormap import sauron

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.benchmark=True
torch.cuda.fastest =True
device = torch.device("cuda")

torch.cuda.empty_cache()

save_directory = '/home/corona/c1307135/Semantic_ML/Corellia/images/2D/'
model_directory = '/home/corona/c1307135/Semantic_ML/Corellia/models/'

#_____________________________________________________________________________#
#_____________________________________________________________________________#
    
def training(model:torch.nn.Module,batch_size,epochs,loss_function,initial_lr,
             save_dir=None,model_dir=None,gradients=False,testing=False):
    
    #_________________________________________________________________________#
    #~~~ SET UP THE MODEL   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #_________________________________________________________________________#
    
    model = model.to(device).to(torch.float)
    
    num_cores = 14
    pool = mp.Pool(num_cores)

    for epoch in range(epochs):
        
        print("Epoch {0} of {1}" .format( (epoch+1), epochs))
        optim = torch.optim.Adam(model.parameters(),learning_rate(initial_lr,epoch))

        #_________________________________________________________________________#
        #~~~ CREATE THE BATCH  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #_________________________________________________________________________#
        
        index = np.arange(0,batch_size,1)
        results = list(pool.imap(return_cube,index))
        batch = np.array([r[0] for r in results])
        targets = np.array([r[1:] for r in results]) 
                      
        batch = torch.tensor(batch)
        batch[batch!=batch]=0
        targets = torch.tensor(targets) 
                
        mom0s = batch.unsqueeze(1) # Resize for B,C,H,W format
        
        targets = targets.squeeze(1).squeeze(-1)
     
        mom0s = mom0s.to(device).to(torch.float)                                # Cast inputs to GPU
        
        targets = targets.to(device).to(torch.float)
        pos_t = targets[:,0]

        #_________________________________________________________________________#
        #~~~ TRAIN THE MODEL   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #_________________________________________________________________________#
        
        model.train(True) 
        
        running_loss = []
        
        for _ in tqdm(range(100)):
            
            prediction, inc = model(mom0s,pos_t)
            loss = loss_function(prediction, mom0s)
            prediction.retain_grad()
            optim.zero_grad(); loss.backward(); optim.step();
            running_loss.append(loss.detach().cpu())
                        
        print("\n Mean loss: %.6f" % np.mean(running_loss),
                  "\t Loss std: %.6f" % np.std(running_loss),
                  "\t Learning rate: %.6f:" % learning_rate(initial_lr,epoch))
        print('_'*73)
                 
        #_________________________________________________________________________#
        #~~~ CREATE ANY PLOTS   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #_________________________________________________________________________#
        
        if (save_dir is not None) and (epoch == epochs-1):
            plotter(prediction, mom0s, inc, save_directory)
            
        #_________________________________________________________________________#
        #~~~ PRINT AVERAGE LAYER GRADIENTS   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #_________________________________________________________________________#
        
        if gradients is True:
            layers = []
            av_grads = []
            for n, p in model.named_parameters():
                if (p.requires_grad) and ("bias" not in n):
                    layers.append(n)
                    av_grads.append(p.grad.abs().mean())
            for i in range(len(layers)):
                print(layers[i],' grad: ',av_grads[i])
                         
        #_________________________________________________________________________#
        #~~~ SAVE THE MODEL   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #_________________________________________________________________________#
        
        if (model_dir is not None) and (epoch == epochs-1):
            torch.save(model.state_dict(),model_dir+'semantic_AE_.pt')  
            
        #_________________________________________________________________________#
        #~~~ SMAKE ENCODINGS FOR DEBUGGING   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #_________________________________________________________________________#
            
        if (epoch == 0) and (testing == True):
            print('TARGETS: ', targets)
            model.train(False)
            encodings = model.test_encode(batch)
            print('ENCODINGS: ',encodings)
        
        del batch; del targets;
        
#_____________________________________________________________________________#
#_____________________________________________________________________________#
                
batch_size = 8

### RUN THE TRAINING PROCEDURE HERE AND PROVIDE THE NETWORK WITH THE REQUIRED
### AUXILIARY 2D X AND Y ARRAYS        

### Create the auxiliary arrays
l = torch.arange(0 - 63/2., (63/2.)+1)
yyy, xxx, zzz = torch.meshgrid(l,l,l)

xxx, yyy, zzz = xxx.repeat(batch_size,1,1,1), yyy.repeat(batch_size,1,1,1), zzz.repeat(batch_size,1,1,1)
xxx = xxx.to(device).to(torch.float)
yyy = yyy.to(device).to(torch.float)
zzz = zzz.to(device).to(torch.float)

mask = torch.zeros((batch_size,1,64,64)).to(device).to(torch.float)
thresh = torch.tensor([0]).to(device).to(torch.float)

#_____________________________________________________________________________#
#_____________________________________________________________________________#

model = CAE(xxx,yyy,zzz,thresh,mask) # Instantiate the model with 6 learnable parameters
print(model)
#torch.nn.MSELoss()
#torch.nn.SmoothL1Loss()

### Train the model
training(model,batch_size=batch_size,epochs=100,loss_function=torch.nn.MSELoss(),
         initial_lr=1e-4,model_dir=model_directory,save_dir=save_directory,gradients=False)

#_____________________________________________________________________________#
#_____________________________________________________________________________#




