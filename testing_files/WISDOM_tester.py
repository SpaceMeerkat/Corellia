#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 14:21:53 2020

@author: SpaceMeerkat

This script takes the WISDOM moment maps and encodes them using a pretrained
network. The encodings are then used to create output images for comparison
to the original moment maps. This is a blind test script and therefore there
is no GPU coding involved.

"""
# =============================================================================
# Import relevant packages
# =============================================================================

import numpy as np
import torch
import sys
sys.path.append("../training_files_2D/")
from model2 import CAE
import pandas as pd
from functions import WISDOM_plotter
from WISDOM_utils import WISDOM_loader as load
import glob

# =============================================================================
# Setup paths
# =============================================================================

model_path = '/home/corona/c1307135/Semantic_ML/Corellia/models/'
data_path = '/home/corona/c1307135/Semantic_ML/Corellia/WISDOM/fits_files/'

# =============================================================================
# Apply dropout to get the mean and std for learned parameters
# =============================================================================

def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train(True)

model = CAE()
model.load_state_dict(torch.load(model_path+'semantic_AE_WISDOM_ONLY.pt'))
model = model.cpu()
model.train(False)
model.apply(apply_dropout)

print("Model cast to CPU")

# =============================================================================
# Load in the WISDOM data
# =============================================================================

# WISDOM_gals = np.array(['NGC0524','NGC4429','NGC0383','NGC4697'])

# =============================================================================
# Collect all WISDOM filenames for testing all of them
# =============================================================================

mom0_filenames = glob.glob(data_path + '*mom0.fits')
mom0_filenames.sort()

names = []
for i in mom0_filenames:
    name = i.split('/')[-1]
    if name[0] == 'N':
        names.append(name[:7])
    else:
        names.append(name[:5])
names = np.array(names)

# =============================================================================
# Collecting the mom0 and mom1 tensors for testing
# =============================================================================

mom0s, mom1s, pos, cdelts, sizes, vcirc = load(data_path = data_path)
    
mom0s, mom1s = np.array(mom0s), np.array(mom1s)

mom0s = torch.tensor(mom0s).to(torch.float)
mom1s = torch.tensor(mom1s).to(torch.float)

pos = torch.zeros(mom0s.shape[0]).to(torch.float)

print("Test data created")

# =============================================================================
# Collecting blind test WISDOM parameters via encoding
# =============================================================================

predictions = []
errors = []
for j in range(mom0s.shape[0]): 
    temp_pred = []
    for _ in range(2000):
        prediction1 = model.test_encode(mom0s[j].unsqueeze(0),mom1s[j].unsqueeze(0),
                                        pos[j].unsqueeze(0))
        prediction1 = prediction1.detach().numpy()
        temp_pred.append(prediction1)
    temp_pred = np.vstack(temp_pred)
    mean_pred = np.mean(temp_pred,0)
    predictions.append(mean_pred)
    errors.append(np.sum(np.abs(temp_pred-mean_pred[None,:]),0)/len(temp_pred))
        
print("Testing data complete")

predictions = np.vstack(predictions)
errors = np.vstack(errors)

dfp = pd.DataFrame(predictions)
dfe = pd.DataFrame(errors)
dfn = pd.DataFrame(names)

df = pd.concat([dfn,dfp,dfe],axis=1)
df.columns = ['OBJECT',0,1,2,3,4,5,6,7,8,9]

# =============================================================================
# Get the scale lengths in arcsecond units
# =============================================================================

# dfp[2] *= cdelts
# dfp[3] *= cdelts

# dfe[2] *= cdelts
# dfe[3] *= cdelts

df.to_pickle('/home/corona/c1307135/Semantic_ML/Corellia/WISDOM/pickle_files/WISDOM.pkl')

# =============================================================================
# Put the medians back into the network to get "out" images
# =============================================================================

predictions = torch.tensor(predictions).to(torch.float).unsqueeze(1).unsqueeze(1).unsqueeze(1)

batch_size = predictions.shape[0]     

### Create the auxiliary arrays
l = torch.arange(0 - 63/2., (63/2.)+1)
yyy, xxx, zzz = torch.meshgrid(l,l,l)

xxx, yyy, zzz = xxx.repeat(batch_size,1,1,1), yyy.repeat(batch_size,1,1,1), zzz.repeat(batch_size,1,1,1)
xxx = xxx.to(torch.float)
yyy = yyy.to(torch.float)
zzz = zzz.to(torch.float)

mom0s[mom0s<0.001] = 0
mom1s[mom0s==0] = 0

BRIGHTNESS, VELOCITY, vmax = CAE(xxx,yyy,zzz).test_images(mom0s, mom1s, predictions[:,:,:,:,0], 
                                                          predictions[:,:,:,:,1], predictions[:,:,:,:,2], 
                                                          predictions[:,:,:,:,3], predictions[:,:,:,:,4],
                                                          predictions[:,:,:,:,0]*0 + 1, shape=64)

for i in range(mom0s.shape[0]):
    WISDOM_plotter(sizes,mom0s.squeeze(1).numpy(),mom1s.squeeze(1).numpy(),
                   BRIGHTNESS.squeeze(1),VELOCITY.squeeze(1),
                   dfp.values,dfe.values, i, vcirc,
                   '/home/corona/c1307135/Semantic_ML/Corellia/WISDOM/Pred_images/'+dfn.iloc[i].values[0]+'.png')
     
# =============================================================================
# End of script
# =============================================================================

