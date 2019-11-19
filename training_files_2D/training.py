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

from functions import learning_rate, return_cube, plotter
from model import CAE

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.benchmark=True
torch.cuda.fastest =True
device = torch.device("cuda")

torch.cuda.empty_cache()

save_directory = '/home/corona/c1307135/Semantic_ML/Corellia/images/'

#_____________________________________________________________________________#
#_____________________________________________________________________________#
    
def training(model:torch.nn.Module,batch_size,epochs,loss_function,initial_lr,
             save_dir=None,gradients=False):
    
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
        targets = np.array([r[1:] for r in results]) # Note: not needed for CAE
                      
        batch = torch.tensor(batch)
        targets = torch.tensor(targets) 
        
        # Resize for B,C,H,W format
        batch = batch.unsqueeze(0)
        batch[batch!=batch]=0
        #batch = batch.permute(0,3,1,2) # reshape the tensor to (B,C,H,W)
        targets = targets.squeeze(1).squeeze(-1)
     
        # Cast inputs to GPU
        
        batch = batch.to(device).to(torch.float)
        targets = targets.to(device).to(torch.float)
               
        #_________________________________________________________________________#
        #~~~ TRAIN THE MODEL   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #_________________________________________________________________________#
        
        model.train(True) 
        
        running_loss = []
        
        for _ in tqdm(range(10000)):
            prediction, rr_t = model(batch)  
            loss = loss_function(prediction, batch)
            prediction.retain_grad()
            optim.zero_grad(); loss.backward(); optim.step();
            running_loss.append(loss.detach().cpu())
            
        print("\n Mean loss: %.8f" % np.mean(running_loss),
                  "\n Loss std: %f" % np.std(running_loss))
        print('_'*73)
        
        #_________________________________________________________________________#
        #~~~ CREATE ANY PLOTS   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #_________________________________________________________________________#
        
        if save_dir is not None:
            plotter(rr_t, batch, prediction, save_directory)
            
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
        
        if (save_dir is not None) and (epoch == 39):
            torch.save(model.state_dict(),save_dir+'CAE_pilot.pt')  
            
        #_________________________________________________________________________#
        #~~~ SMAKE ENCODINGS FOR DEBUGGING   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #_________________________________________________________________________#
            
        if epoch == 0 :
            print('TARGETS: ', targets)
            model.train(False)
            encodings = model.test_encode(batch)
            print('ENCODINGS: ',encodings)
        
        del batch; del targets;
        
#_____________________________________________________________________________#
#_____________________________________________________________________________#
        
        
### RUN THE TRAINING PROCEDURE HERE AND PROVIDE THE NETWORK WITH THE REQUIRED
### AUXILIARY 2D X AND Y ARRAYS        

### Create the auxiliary arrays
yy, xx = torch.meshgrid(torch.arange(0 - 63/2., (63/2.)+1), torch.arange(0 - 63/2., (63/2.)+1))
xx = xx.to(device).to(torch.float)
yy = yy.to(device).to(torch.float)

cube = torch.zeros((120,64,64)).to(device).to(torch.float)

model = CAE(4,xx,yy,cube) # Instantiate the model with 6 learnable parameters

### Train the model
training(model,batch_size=1,epochs=1,loss_function=torch.nn.MSELoss(),
         initial_lr=1e-5,save_dir=save_directory,gradients=True)






