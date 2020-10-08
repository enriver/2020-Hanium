# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 23:14:38 2020

@author: seungjun
"""

import matplotlib.pyplot as plt
import matplotlib as mpl 
import matplotlib.font_manager as fm

import pandas as pd
import numpy as np
import os 
import random

from pathlib import Path
from shutil import copyfile, move


### torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torch
import torchvision as tv
from torch.utils.data import Dataset




def single_stock_generator(chart, labels, batch, dimension):
    #output = [chart, labels]
    transform = tv.transforms.Compose([tv.transforms.ToPILImage(mode = 'RGB'),
    tv.transforms.Resize(dimension),
    tv.transforms.ToTensor(),
    #tv.transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
    ])
    num = len(labels)//batch
    for j in range(num):
        stock_batch = np.zeros(shape = (batch, 3, dimension, dimension))
        label_batch = np.zeros(shape=(batch,))
        idx=j*batch
        for i in range(batch):
            
            #idx = np.random.randint(len(labels))
            kkk= transform(chart[idx])
            kk=kkk.numpy()

            stock_batch[i]=kk
            label_batch[i]=labels[idx]
            idx+=1
        stock_batch = torch.tensor(stock_batch).float()
        label_batch = torch.tensor(label_batch).long()
        yield stock_batch, label_batch


def single_stock_valid_generator(chart, labels, batch, dimension):
    #output = [chart, labels]
    transform = tv.transforms.Compose([tv.transforms.ToPILImage(mode = 'RGB'),
    tv.transforms.Resize(dimension),
    tv.transforms.ToTensor(),
    #tv.transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
    ])
    num = len(labels)//batch
    for j in range(num):
        stock_batch = np.zeros(shape = (batch, 3, dimension, dimension))
        label_batch = np.zeros(shape=(batch,))
        idx=j*batch
        for i in range(batch):
            
            #idx = np.random.randint(len(labels))
            kkk= transform(chart[idx])
            kk=kkk.numpy()

            stock_batch[i]=kk
            label_batch[i]=labels[idx]
            idx+=1
        stock_batch = torch.tensor(stock_batch).float()
        label_batch = torch.tensor(label_batch).long()
        yield stock_batch, label_batch
        
        
class Pathdataset(Dataset):
    def __init__(self, image, labels = None, test_mode = True, transform = None):
        self.len = len(image)
        self.image = image
        self.labels = labels
        self.mode = test_mode
        self.transform = transform
        
    def __getitem__(self, index):
        im = self.image[index]
        im = self.transform(im)
        
        
        if self.mode:
            #valid
            im=self.transform(im)
            
            return im
        
        else:
            #train
            im = self.transform(im)

            return im,\
                 torch.tensor(self.labels[index] ,dtype=torch.long)
        
    def __len__(self):
        return self.len
            
            

def FBdataset(Dataset):
    data = pd.DataFrame(
        { 'ds': Dataset.index,         
          'y' : Dateset['Close']})
    data['cap']=data.y.max()      
    data['floor']=data.y.min()
    data.reset_index( inplace=True )
    del data['Date']
    data=data.fillna(method='ffill')
    return data