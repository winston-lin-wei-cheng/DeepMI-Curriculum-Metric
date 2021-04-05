#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:22:39 2019

@author: winston
"""
import pandas as pd
import numpy as np
import os
import pickle
import torch
import random
from torch.utils.data.sampler import Sampler
import models

# Ignore warnings & Fix random seed
import warnings
warnings.filterwarnings("ignore")
random.seed(999)
seed=99


def getPaths_unlabel_Podcast(path_podcast, sample_num=90000, shuffle=True):
    """
    This function random selects unlabeled samples for SSL-DeepEmoCluster
    Args:
        path_podcast$ (str): path of the unlabeled dataset
        sample_num$ (int): number of samples to be selected
    """

    for dirPath, dirNames, fileNames in os.walk(path_podcast):
        # randomly pick unlabeled utterances
        fileNames = sorted(fileNames)
        indices = list(range(len(fileNames)))
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
        indices = indices[:sample_num]
        indices = np.array(indices)
        fileNames = list(np.array(fileNames)[indices].astype('str'))  
        whole_fnames = []
        for i in range(len(fileNames)):
            whole_fnames.append(fileNames[i].replace('.mat','.wav'))
        whole_fnames = np.array(whole_fnames).astype('str')
        return whole_fnames

def getPaths_attri(path_label, split_set, emo_attr):
    """
    This function is for filtering data by different constraints of label
    Args:
        path_label$ (str): path of label
        split_set$ (str): 'Train', 'Validation' or 'Test' 
        emo_attr$ (str): 'Act', 'Dom' or 'Val'
    """
    label_table = pd.read_csv(path_label)
    whole_fnames = (label_table['FileName'].values).astype('str')
    split_sets = (label_table['Split_Set'].values).astype('str')
    emotions = label_table['Emo'+emo_attr].values
    _paths = []
    _label_emo = []
    for i in range(len(whole_fnames)):
        # Constrain with Split Sets      
        if split_sets[i]==split_set:
            # Constrain with Emotional Labels
            _paths.append(whole_fnames[i])
            _label_emo.append(emotions[i])
        else:
            pass
    return np.array(_paths), np.array(_label_emo)

def cc_coef(output, target):
    mu_y_true = torch.mean(target)
    mu_y_pred = torch.mean(output)                                                                                                                                                                                              
    return 1 - 2 * torch.mean((target - mu_y_true) * (output - mu_y_pred)) / (torch.var(target) + torch.var(output) + torch.mean((mu_y_pred - mu_y_true)**2))

def evaluation_metrics(true_value,predicted_value):
    corr_coeff = np.corrcoef(true_value,predicted_value)
    ccc = 2*predicted_value.std()*true_value.std()*corr_coeff[0,1]/(predicted_value.var() + true_value.var() + (predicted_value.mean() - true_value.mean())**2)
    return(ccc,corr_coeff)

class Logger(object):
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)

class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_model(path, num_clusters):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        model = models.__dict__[checkpoint['arch']](inp=6373, out=num_clusters)

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model
