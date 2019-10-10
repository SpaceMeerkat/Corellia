#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:00:38 2019

@author: SpaceMeerkat
"""

#_____________________________________________________________________________#
#_____________________________________________________________________________#

import torch
from cube_generator import cube_generator
from model import CovNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.benchmark=True
torch.cuda.fastest =True

#_____________________________________________________________________________#
#_____________________________________________________________________________#

def learning_rate(initial_lr, epoch):
    """Sets the learning rate to the initial LR decayed by a factor of 10 every
    N epochs"""
    lr = initial_lr * (0.9 ** (epoch// 1))
    return lr 

#_____________________________________________________________________________#
#_____________________________________________________________________________#

f = cube_generator(scale_length_frac=4,pos_ang=45,inc_ang=70,resolution=1000).cube_creation()

