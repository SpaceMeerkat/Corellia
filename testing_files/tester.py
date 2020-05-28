#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:33:59 2019

@author: SpaceMeerkat
"""
#_____________________________________________________________________________#
#_____________________________________________________________________________#

import numpy as np
import torch
import multiprocessing as mp
import sys
sys.path.append("../training_files_2D/")
from model2 import CAE
from functions import return_cube
import pandas as pd

#_____________________________________________________________________________#
#_____________________________________________________________________________#

model_path = '/home/corona/c1307135/Semantic_ML/Corellia/models/'

#_____________________________________________________________________________#
#_____________________________________________________________________________#

model = CAE()
model.load_state_dict(torch.load(model_path+'semantic_AE_.pt'))
model = model.cpu()
model.train(False)
print("Model cast to CPU")

#_____________________________________________________________________________#
#_____________________________________________________________________________#

index = np.arange(0,100,1)
num_cores = 14
pool = mp.Pool(num_cores,maxtasksperchild=100)
results = list(pool.imap(return_cube,index))
pool.close()

batch = np.array([r[0] for r in results])
target = np.array([r[1:] for r in results])

batch = torch.tensor(batch).to(torch.float)
batch[batch!=batch]=0
target = torch.tensor(target).to(torch.float)

pos = target[:,0]
ah = target[:,-2]

mom0s, mom1s = batch[:,0,:,:].unsqueeze(1), batch[:,1,:,:].unsqueeze(1)

print("Test data created")

#_____________________________________________________________________________#
#_____________________________________________________________________________#

predictions = []
for j in range(batch.shape[0]):    
    prediction1 = model.test_encode(mom0s[j].unsqueeze(0),mom1s[j].unsqueeze(0),
                                    pos[j].unsqueeze(0))#, ah[j].unsqueeze(0))
    prediction1 = prediction1.detach().numpy()
    predictions.append(prediction1)
        
print("Testing data complete")

predictions = np.vstack(predictions)

dfp = pd.DataFrame(predictions)
dft = pd.DataFrame(np.vstack(target))

df = pd.concat([dfp,dft],axis=1)
df.columns = [0,1,2,3,4,5,6,7,8,9]

df.to_pickle('/home/corona/c1307135/Semantic_ML/Corellia/pickle_files/'+str(np.random.uniform(0,1,1)[0])+'.pkl')

#_____________________________________________________________________________#
#_____________________________________________________________________________#


