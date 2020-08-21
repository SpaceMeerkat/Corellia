#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:00:38 2019

@author: SpaceMeerkat

A training script for the Self-supervised Physics aware Neural Network. The 
script generates moment maps using the KinMS package and learns the needed
parameters for reconstructing moment maps. 

This script uses PyTorch's CUDA integrated functionality for GPU accelerated
training.
"""

# =============================================================================
# Import relevant packages
# =============================================================================

import multiprocessing as mp
import numpy as np
import os
from tqdm import tqdm
import torch
import sys
sys.path.append("../testing_files/")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from functions import learning_rate, return_cube, plotter
from model2 import CAE
from cube_generator import cube_generator
from sauron_colormap import sauron
from WISDOM_utils import WISDOM_loader as load

# =============================================================================
# Setup the GPU CUDA functionality
# =============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.benchmark=True
torch.cuda.fastest =True
device = torch.device("cuda")

torch.cuda.empty_cache()

# =============================================================================
# Setup the neccessary paths 
# =============================================================================

save_directory = '/home/corona/c1307135/Semantic_ML/Corellia/images/2D/'
model_directory = '/home/corona/c1307135/Semantic_ML/Corellia/models/'
WISDOM_path = '/home/corona/c1307135/Semantic_ML/Corellia/WISDOM/fits_files/'

# =============================================================================
# Define the training function
# =============================================================================
    
def training(model:torch.nn.Module,batch_size,epochs,loss_function,initial_lr,
             save_dir=None,model_dir=None,WISDOM_path=None,gradients=False,
             testing=False):
    
# =============================================================================
#   Setup the model for training on the GPU
# =============================================================================
    
    model = model.to(device).to(torch.float)
    
    num_cores = 14
    pool = mp.Pool(num_cores)
    
# =============================================================================
#   Load in the WISDOM data if needed for training
# =============================================================================
    
    if WISDOM_path is not None:
        Wmom0s, Wmom1s, Wpos, cdelts, sizes, vcircs = load(data_path = WISDOM_path)
        Wmom0s = Wmom0s.to(device).to(torch.float)
        Wmom1s = Wmom1s.to(device).to(torch.float)
        Wpos = Wpos.to(device).to(torch.float)
        

    for epoch in range(epochs):
        
        model.train(True) 
        
        running_loss = []
        
        print("Epoch {0} of {1}" .format( (epoch+1), epochs))
        optim = torch.optim.Adam(model.parameters(),learning_rate(initial_lr,epoch))
        
# =============================================================================
#       (Optional) push through the WISDOM data every 5th epoch
# =============================================================================
        
        if (WISDOM_path is not None) and (epoch % 1 == 0):
            
            print('\n', '='*73, "\n [>>>   Pushing WISDOM data through the network   <<<]",
                  '\n', '='*73)
            
            for _ in tqdm(range(100)):
                
                prediction1, prediction2, inc, pos, vmax = model(Wmom0s,Wmom1s,Wpos)
                loss1 = loss_function(prediction1, Wmom0s)
                loss2 = loss_function(prediction2, Wmom1s)
                loss = loss1 + loss2
                prediction1.retain_grad()
                prediction2.retain_grad()
                optim.zero_grad(); loss.backward(); optim.step();
                running_loss.append(loss.detach().cpu())
                
# =============================================================================
#       Create the mini batches for pushing through the network
# =============================================================================
                
        else:
        
            index = np.arange(0,batch_size,1)
            results = list(pool.imap(return_cube,index))
            batch = np.array([r[0] for r in results])
            targets = np.array([r[1:] for r in results]) 
                          
            batch = torch.tensor(batch)
            batch[batch!=batch]=0
            targets = torch.tensor(targets) 
            
            mom0s, mom1s = batch[:,0,:,:].unsqueeze(1), batch[:,1,:,:].unsqueeze(1) # Resize for B,C,H,W format
            
            targets = targets.squeeze(1).squeeze(-1)
         
            mom0s = mom0s.to(device).to(torch.float)                                # Cast inputs to GPU
            mom1s = mom1s.to(device).to(torch.float)
            
            targets = targets.to(device).to(torch.float)
                    
            pos_t = targets[:,0]
                        
            for _ in tqdm(range(100)):
                
                prediction1, prediction2, inc, pos, vmax = model(mom0s,mom1s,pos_t)
                loss1 = loss_function(prediction1, mom0s)
                loss2 = loss_function(prediction2, mom1s)
                loss = loss1 + loss2
                prediction1.retain_grad()
                prediction2.retain_grad()
                optim.zero_grad(); loss.backward(); optim.step();
                running_loss.append(loss.detach().cpu())
                
            del batch; del targets;
                
# =============================================================================
#       Print the running average loss and spread
# =============================================================================
                                    
        print("\n Mean loss: %.6f" % np.mean(running_loss),
                  "\t Loss std: %.6f" % np.std(running_loss),
                  "\t Learning rate: %.6f:" % learning_rate(initial_lr,epoch))
        print('_'*73)
                    
# =============================================================================
#       (Optional) print the layer gradients
# =============================================================================
        
        if gradients is True:
            layers = []
            av_grads = []
            for n, p in model.named_parameters():
                if (p.requires_grad) and ("bias" not in n):
                    layers.append(n)
                    av_grads.append(p.grad.abs().mean())
            for i in range(len(layers)):
                print(layers[i],' grad: ',av_grads[i])
                         
# =============================================================================
#       (Optional) save the model's state dictionary
# =============================================================================
        
        if (model_dir is not None) and (epoch == epochs-1):
            torch.save(model.state_dict(),model_dir+'semantic_AE_WISDOM_ONLY.pt')  
            
# =============================================================================
#       (Optional) create plots
# =============================================================================
        
        if (save_dir is not None) and (epoch == epochs-1):
            plotter(prediction1, prediction2, Wmom0s, Wmom1s, inc, pos, save_directory)
            
# =============================================================================
#       (Optional) test mode to check embeddings work and there are no bugs
# =============================================================================
            
        if (epoch == 0) and (testing == True):
            print('TARGETS: ', targets)
            model.train(False)
            encodings = model.test_encode(batch)
            print('ENCODINGS: ',encodings)
        
# =============================================================================
# Setup the auxilliary tensors for the training run
# =============================================================================
                
batch_size = 64

l = torch.arange(0 - 63/2., (63/2.)+1)
yyy, xxx, zzz = torch.meshgrid(l,l,l)

xxx, yyy, zzz = xxx.repeat(batch_size,1,1,1), yyy.repeat(batch_size,1,1,1), zzz.repeat(batch_size,1,1,1)
xxx = xxx.to(device).to(torch.float)
yyy = yyy.to(device).to(torch.float)
zzz = zzz.to(device).to(torch.float)

mask = torch.zeros((xxx.shape)).to(device).to(torch.float)
thresh = torch.tensor([3]).to(device).to(torch.float)

# =============================================================================
# Instantiate the model
# =============================================================================

model = CAE(xxx,yyy,zzz,thresh,mask) # Instantiate the model with 6 learnable parameters
print(model)

# =============================================================================
# Kickstart the training 
# =============================================================================

training(model,batch_size=batch_size,epochs=300,loss_function=torch.nn.MSELoss(),
         initial_lr=1e-4,model_dir=model_directory,save_dir=save_directory,
         WISDOM_path = WISDOM_path, gradients=False)

# =============================================================================
# End of script
# =============================================================================



