#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:09:05 2019

@author: winstonlin
"""
import torch
import torch.nn as nn
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import MspPodcast_CurriculumTrain, MspPodcast_Validation
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import MultiStepLR
from utils import cc_coef
import argparse
torch.manual_seed(1)



# simple DenseNet emotional regressor
class DenseNet(torch.nn.Module): 
    def __init__(self, input_dim, output_dim):
        super(DenseNet, self).__init__()
        
        # STL-Dense-layers
        self.dense_net = nn.Sequential(nn.Linear(input_dim, 512),
                                    nn.Dropout(0.3),
                                    nn.Linear(512, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.3),
                                    nn.Linear(512, 512),
                                    nn.ReLU(),                                   
                                    nn.Linear(512, output_dim))                   
        
    def forward(self, x):
        y = self.dense_net(x)
        return y

def train(loader, model, crit, opt):
    model.train()
    training_loss = []
    for i, data_batch in enumerate(tqdm(loader)):
        x, y = data_batch
        x = x.float().cuda()
        y = y.view(y.size(0), 1) # match shape for the CCC
        y = y.float().cuda()     # loss calculation (important notice!)
        # model flow and loss computation
        pred = model(x)
        loss = crit(pred, y)
        # update weights
        opt.zero_grad()     # clear gradients for this training step
        loss.backward()     # backpropagation, compute gradients
        opt.step()          # apply gradients
        # record loss
        training_loss.append(loss.data.cpu().numpy()) 
    return np.mean(training_loss)
    
def validation(loader, model, crit):
    model.eval()
    validation_loss = []
    for i, data_batch in enumerate(tqdm(loader)):
        x, y = data_batch
        x = x.float().cuda()
        y = y.view(y.size(0), 1) # match shape for the CCC
        y = y.float().cuda()     # loss calculation (important notice!)
        # model flow and loss computation
        pred = model(x)
        loss = crit(pred, y)
        # record loss
        validation_loss.append(loss.data.cpu().numpy())    
    return np.mean(validation_loss)
###############################################################################


argparse = argparse.ArgumentParser()
argparse.add_argument("-ep", "--epoch", required=True)
argparse.add_argument("-batch", "--batch_size", required=True)
argparse.add_argument("-emo", "--emo_attr", required=True)
args = vars(argparse.parse_args())

# Dirs & Parameters
root_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.8/Features/OpenSmile_func_IS13ComParE/feat_mat/'
label_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.8/Labels/labels_concensus.csv'

batch_size = int(args['batch_size'])
epochs = int(args['epoch'])
emo_attr = args['emo_attr']

# creating checkpoint repo
exp = './Models/'
if not os.path.isdir(exp):
    os.makedirs(exp)

# Model Saving Path
SAVING_PATH = './Models/SimpleEmoRegressor_DeepMI@_'+emo_attr+'.pt.tar'

# loading Curriculum Metric 
# JSON_PATH = './curriculum_metric/DeepMI_50clusters@_'+emo_attr+'.json'
JSON_PATH = './curriculum_metric_v1.8/DeepMI_50clusters@_'+emo_attr+'.json'
with open(JSON_PATH) as fh:
    metric = json.load(fh)   

# loading model structures
model = DenseNet(input_dim=6373, output_dim=1)
model.cuda()

# PyTorch Dataset Wrapper
level_1_difficulty = MspPodcast_CurriculumTrain(root_dir, label_dir, emo_attr, metric, difficulty_level=1)
level_2_difficulty = MspPodcast_CurriculumTrain(root_dir, label_dir, emo_attr, metric, difficulty_level=2)
level_3_difficulty = MspPodcast_CurriculumTrain(root_dir, label_dir, emo_attr, metric, difficulty_level=3)
level_4_difficulty = MspPodcast_CurriculumTrain(root_dir, label_dir, emo_attr, metric, difficulty_level=4)
level_5_difficulty = MspPodcast_CurriculumTrain(root_dir, label_dir, emo_attr, metric, difficulty_level=5)
level_6_difficulty = MspPodcast_CurriculumTrain(root_dir, label_dir, emo_attr, metric, difficulty_level=6)
level_7_difficulty = MspPodcast_CurriculumTrain(root_dir, label_dir, emo_attr, metric, difficulty_level=7)
level_8_difficulty = MspPodcast_CurriculumTrain(root_dir, label_dir, emo_attr, metric, difficulty_level=8)
level_9_difficulty = MspPodcast_CurriculumTrain(root_dir, label_dir, emo_attr, metric, difficulty_level=9)
level_10_difficulty = MspPodcast_CurriculumTrain(root_dir, label_dir, emo_attr, metric, difficulty_level=10)
    
# Creating validation data samplers and loaders:
validation_dataset = MspPodcast_Validation(root_dir, label_dir, emo_attr)
valid_sampler = SubsetRandomSampler(list(range(len(validation_dataset))))
valid_loader = DataLoader(validation_dataset, 
                          batch_size=batch_size,
                          sampler=valid_sampler,
                          num_workers=12,
                          pin_memory=True)

# Model Training Settings: loss_function, optimizer, learning_rate
optimizer = optim.Adam(model.parameters(), lr=0.001) 
scheduler = MultiStepLR(optimizer, milestones=[4,9,14,19,24,29,34,39,44], gamma=0.5)

# Train and Validation
Epoch_training_Loss = []
Epoch_validation_Loss = []
val_loss = 0
for epoch in range(epochs):         
    # perform curriculum training 
    if epoch<5:
        training_dataset = level_1_difficulty
    elif (epoch>=5)&(epoch<10):
        training_dataset = level_2_difficulty
    elif (epoch>=10)&(epoch<15):
        training_dataset = level_3_difficulty
    elif (epoch>=15)&(epoch<20):
        training_dataset = level_4_difficulty
    elif (epoch>=20)&(epoch<25):
        training_dataset = level_5_difficulty            
    elif (epoch>=25)&(epoch<30):
        training_dataset = level_6_difficulty            
    elif (epoch>=30)&(epoch<35):
        training_dataset = level_7_difficulty            
    elif (epoch>=35)&(epoch<40):
        training_dataset = level_8_difficulty            
    elif (epoch>=40)&(epoch<45):
        training_dataset = level_9_difficulty             
    else:
        training_dataset = level_10_difficulty             
    
    # creating training samplers and loaders
    print('Size of Training Set'+str(len(training_dataset))+'\n')
    train_sampler = SubsetRandomSampler(list(range(len(training_dataset))))
    train_loader = DataLoader(training_dataset, 
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=12,
                              pin_memory=True)    
    
    # train/valid process
    loss_train = train(train_loader, model, cc_coef, optimizer)
    loss_valid = validation(valid_loader, model, cc_coef)   
    scheduler.step()
    Epoch_training_Loss.append(loss_train)
    Epoch_validation_Loss.append(loss_valid)
           
    # Record/Report Epoch-loss
    print('Epoch: '+str(epoch)+' ,Training-loss: '+str(loss_train)+' ,Validation-loss: '+str(loss_valid))
        
    # Checkpoint for saving best Model based on val-loss
    if epoch==0:
        val_loss = loss_valid
        torch.save(model.state_dict(), SAVING_PATH)
        print("=> Saving the initial best model (Epoch="+str(epoch)+")")
    else:
        if val_loss > loss_valid:
            torch.save(model.state_dict(), SAVING_PATH)
            print("=> Saving a new best model (Epoch="+str(epoch)+")")
            print("=> Loss reduction from "+str(val_loss)+" to "+str(loss_valid))
            val_loss = loss_valid
        else:
            print("=> Validation Loss did not improve (Epoch="+str(epoch)+")")
    print('=================================================================')

# Drawing Loss/Acc Curve for Epoch-based and Batch-based
plt.title('Epoch-Loss Curve')
plt.plot(Epoch_training_Loss,color='blue',linewidth=4)
plt.plot(Epoch_validation_Loss,color='red',linewidth=4)
plt.savefig(SAVING_PATH.replace('.pt.tar','_Epoch.png'))
