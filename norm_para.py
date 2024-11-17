#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
import os
import numpy as np
from scipy.io import loadmat, savemat
import random
from utils import getPaths_attri

# Ignore warnings & Fix random seed
import warnings
warnings.filterwarnings("ignore")
random.seed(999)



if __name__=='__main__':

    # checking/creating output directory
    if not os.path.exists('./NormTerm'):
        os.mkdir('./NormTerm/')
    
    # compute z-norm parameters
    emotions = ['Act', 'Dom', 'Val']
    for emo_attr in emotions:
        data_root = '/YOUR/PATH/TO/MSP-PODCAST-Publish-1.8/Features/OpenSmile_func_IS13ComParE/feat_mat/'
        label_path = '/YOUR/PATH/TO/MSP-PODCAST-Publish-1.8/Labels/labels_concensus.csv'
        fnames, Train_Label = getPaths_attri(label_path, split_set='Train', emo_attr=emo_attr)
    
        # Acoustic-Feature Normalization based on Training Set
        Train_Data = []
        for i in range(len(fnames)):
            data = loadmat(data_root + fnames[i].replace('.wav','.mat'))['Audio_data']
            data = data.reshape(-1)
            Train_Data.append(data)
        Train_Data = np.array(Train_Data)
    
        # Feature Normalization Parameters
        Feat_mean_All = np.mean(Train_Data,axis=0)
        Feat_std_All = np.std(Train_Data,axis=0)
        savemat('./NormTerm/feat_norm_means.mat', {'normal_para':Feat_mean_All})
        savemat('./NormTerm/feat_norm_stds.mat', {'normal_para':Feat_std_All})
        
        Label_mean = np.mean(Train_Label)
        Label_std = np.std(Train_Label)
        if emo_attr == 'Act':
            savemat('./NormTerm/act_norm_means.mat', {'normal_para':Label_mean})
            savemat('./NormTerm/act_norm_stds.mat', {'normal_para':Label_std})
            
        elif emo_attr == 'Dom':
            savemat('./NormTerm/dom_norm_means.mat', {'normal_para':Label_mean})
            savemat('./NormTerm/dom_norm_stds.mat', {'normal_para':Label_std})
            
        elif emo_attr == 'Val':
            savemat('./NormTerm/val_norm_means.mat', {'normal_para':Label_mean})
            savemat('./NormTerm/val_norm_stds.mat', {'normal_para':Label_std})

