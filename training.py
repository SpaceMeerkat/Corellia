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
import matplotlib.pyplot as plt

from cube_generator import cube_generator
from model import CAE
from functions import learning_rate, return_cube, pos_loss, recover_pos

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.benchmark=True
torch.cuda.fastest =True
device = torch.device("cuda")

torch.cuda.empty_cache()

#_____________________________________________________________________________#
#_____________________________________________________________________________#
    
def training(model:torch.nn.Module,batch_size,epochs,loss_function,initial_lr):
    
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
        
        print(targets.shape)
        print('Theta: ', recover_pos(targets[0,0],targets[0,1]))
        print('Vh: ',targets[0,-1])
               
        batch = torch.tensor(batch)
        targets = torch.tensor(targets)
        
        # Resize for B,C,H,W format
               
        batch = batch.permute(0,3,1,2) # reshape the tensor to (B,C,H,W)
        targets = targets.squeeze(1).squeeze(-1)
     
        # Cast inputs to GPU
        
        batch = batch.to(device).to(torch.float)
        targets = targets.to(device).to(torch.float)
               
        #_________________________________________________________________________#
        #~~~ TRAIN THE MODEL   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #_________________________________________________________________________#
        
        model.train(True) 
        
        running_loss = []
        
#        if epoch == 0:
#            batch_cpu = batch.detach().cpu()
#            for i in range(10):
#                cube = batch_cpu[i,:,:,:]
#                plt.figure()
#                plt.imshow(torch.sum(cube,dim=0),cmap='inferno')
#                plt.savefig('/home/corona/c1307135/Semantic_ML/input_images/'+str(i)+'.png')
#        break
        
        for _ in tqdm(range(10)):
            prediction, vel = model(batch)  
            loss = loss_function(prediction, batch)
            optim.zero_grad(); loss.backward(); optim.step();
            running_loss.append(loss.detach().cpu())
            
            prediction = prediction.detach().cpu().numpy()
            vel = vel.detach().cpu().numpy()
            plt.figure()
            plt.imshow(vel)
            plt.colorbar()
            plt.savefig('/home/corona/c1307135/Semantic_ML/input_images/mom0_'+str(_)+'.png')
                   
        print("\n Mean loss: %.8f" % np.mean(running_loss),
                  "\n Loss std: %f" % np.std(running_loss))
        print('_'*73)
        
        batch = batch.detach().cpu().numpy()
        b = batch[0,:,:,:]
        b[b<np.std(b)*3]=0
        mom0 = b.sum(axis=0)
        channels = np.arange(-60,60,1)
        num = (b.T*channels).sum(axis=0)
        mom1 = num/mom0
        plt.figure()
        plt.imshow(mom1)
        plt.colorbar()
        plt.savefig('/home/corona/c1307135/Semantic_ML/input_images/mom01.png')
        
        #_________________________________________________________________________#
        #~~~ TEST THE MODEL   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #_________________________________________________________________________#
        
        if epoch == 39:
            
            torch.save(model.state_dict(),'/home/corona/c1307135/Semantic_ML/Corellia/'+'CovNet_pilot4.pt')  
        
        del batch; del targets;
        
#_____________________________________________________________________________#
#_____________________________________________________________________________#

yy, xx = torch.meshgrid(torch.arange(0 - 63/2., (63/2.)+1), torch.arange(0 - 63/2., (63/2.)+1))
xx = xx.to(device).to(torch.float)
yy = yy.to(device).to(torch.float)

model = CAE(6,xx,yy)
training(model,batch_size=1,epochs=1,loss_function=torch.nn.MSELoss(),initial_lr=1e-3)






