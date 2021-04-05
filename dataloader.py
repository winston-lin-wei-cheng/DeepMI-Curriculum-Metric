#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:09:05 2019

@author: winston
"""
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from utils import getPaths_attri, getPaths_unlabel_Podcast
import random

# Ignore warnings & Fix random seed
import warnings
warnings.filterwarnings("ignore")
random.seed(999)


###############################################################################
#             For the building of SSL-DeepEmoCluster Dataloaders              #
###############################################################################
class UnlabelDataset(Dataset):
    """ Unlabeled Dataset"""

    def __init__(self, unlabel_podcast_dir):
        # init parameters
        self.unlabel_podcast_dir = unlabel_podcast_dir

        # norm-parameters
        self.Feat_mean = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
        self.Feat_std = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para'] 

        # unlabeled data paths
        self._paths = getPaths_unlabel_Podcast(unlabel_podcast_dir)
        # data path of each utterance
        self.imgs = []
        repeat_paths = self._paths.tolist()
        for i in range(len(repeat_paths)):
            self.imgs.extend([unlabel_podcast_dir+repeat_paths[i]])   
    
    def __len__(self):
        return len(self._paths)
    
    def __getitem__(self, idx):
        # Loading Data MSP-Podcast Unlabeled set
        data = loadmat(self.unlabel_podcast_dir + self._paths[idx].replace('.wav','.mat'))['Audio_data']
        # Z-normalization
        data = (data-self.Feat_mean)/self.Feat_std
        data = data.reshape(-1)
        # Bounded NormFeat Range -3~3 and assign NaN to 0
        data[np.isnan(data)]=0
        data[data>3]=3
        data[data<-3]=-3        
        return data   

class MspPodcastEmoDataset(Dataset):
    """MSP-Podcast Dataset (labeled set)"""

    def __init__(self, root_dir, label_dir, split_set, emo_attr):
        # init parameters
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.split_set = split_set

        # label data paths
        self._paths, self._labels = getPaths_attri(label_dir, split_set, emo_attr)

        # norm-parameters
        self.Feat_mean = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
        self.Feat_std = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']  
        if emo_attr == 'Act':
            self.Label_mean = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0]
        elif emo_attr == 'Dom':
            self.Label_mean = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]
        elif emo_attr == 'Val':
            self.Label_mean = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0]   
        
        # data path/label of each utterance
        self.imgs = []
        repeat_paths = self._paths.tolist()
        repeat_labels = ((self._labels-self.Label_mean)/self.Label_std).tolist()
        for i in range(len(repeat_paths)):
            self.imgs.extend([(root_dir+repeat_paths[i], repeat_labels[i])])
  
    def __len__(self):
        return len(self._paths)
    
    def __getitem__(self, idx):
        # Loading Labeled Data
        data = loadmat(self.root_dir + self._paths[idx].replace('.wav','.mat'))['Audio_data'] 
        # Z-normalization
        data = (data-self.Feat_mean)/self.Feat_std
        data = data.reshape(-1)
        # Bounded NormFeat Range -3~3 and assign NaN to 0
        data[np.isnan(data)]=0
        data[data>3]=3
        data[data<-3]=-3        
        # Loading Label & Normalization
        label = self._labels[idx]
        label = (label-self.Label_mean)/self.Label_std
        return data, label



###############################################################################
#            For the building of Curriculum Learning Dataloaders              #
###############################################################################
def parse_dict_to_arry(metric_dict):
    """This function parses the saved DeepMI curriculum metric to perform curriculum learning"""
    FileName = []
    Metric_Value = []
    for key in metric_dict.keys():
        FileName.append(key)
        Metric_Value.append(metric_dict[key])
    FileName = np.array(FileName)
    Metric_Value = np.array(Metric_Value)
    # sorted by DeepMI-values
    sort_idx = Metric_Value.argsort()
    sort_FileName = FileName[sort_idx[::-1]]
    sort_Metric_Value = Metric_Value[sort_idx[::-1]]
    return sort_FileName, sort_Metric_Value

class MspPodcast_CurriculumTrain(Dataset):
    """MSP-Podcast dataset (for curriculum learning)"""

    def __init__(self, root_dir, label_dir, emo_attr, curriculum_metric, difficulty_level):
        # init parameters
        self.root_dir = root_dir
        
        ##############################################################################
        #   Dynamically define curriculum dataset based on given difficulty metric   #
        ##############################################################################
        # loading all labeled data paths (training set)
        _paths_all, _labels_all = getPaths_attri(label_dir, split_set='Train', emo_attr=emo_attr)

        # sorting by DeepMI curriculum_metric 
        _metric_sort_files, _ = parse_dict_to_arry(curriculum_metric)
        bin_length = int((difficulty_level/10)*len(_paths_all)) # divide into 10-bins of difficulty
        training_files = _metric_sort_files[:bin_length]
        training_labels = []
        for f in training_files:
            training_labels.append(_labels_all[f==_paths_all][0])
        training_labels = np.array(training_labels)
        self._paths = training_files
        self._labels = training_labels
        ###############################################################################
       
        # norm-parameters
        self.Feat_mean = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
        self.Feat_std = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']  
        if emo_attr == 'Act':
            self.Label_mean = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0]
        elif emo_attr == 'Dom':
            self.Label_mean = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]
        elif emo_attr == 'Val':
            self.Label_mean = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0]   
  
    def __len__(self):
        return len(self._paths)
    
    def __getitem__(self, idx):
        # Loading Data
        data = loadmat(self.root_dir + self._paths[idx].replace('.wav','.mat'))['Audio_data'] 
        # Z-normalization
        data = (data-self.Feat_mean)/self.Feat_std
        data = data.reshape(-1)
        # Bounded NormFeat Range -3~3 and assign NaN to 0
        data[np.isnan(data)]=0
        data[data>3]=3
        data[data<-3]=-3        
        # Loading Label & Normalization
        label = self._labels[idx]
        label = (label-self.Label_mean)/self.Label_std
        return data, label

class MspPodcast_Validation(Dataset):
    """MSP-Podcast dataset (for validation)"""

    def __init__(self, root_dir, label_dir, emo_attr):
        # init parameters
        self.root_dir = root_dir
        
        # labeled data paths (validation/development set)
        self._paths, self._labels = getPaths_attri(label_dir, split_set='Validation', emo_attr=emo_attr)            

        # norm-parameters
        self.Feat_mean = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
        self.Feat_std = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']  
        if emo_attr == 'Act':
            self.Label_mean = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0]
        elif emo_attr == 'Dom':
            self.Label_mean = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]
        elif emo_attr == 'Val':
            self.Label_mean = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0]   
  
    def __len__(self):
        return len(self._paths)
    
    def __getitem__(self, idx):
        # Loading Data
        data = loadmat(self.root_dir + self._paths[idx].replace('.wav','.mat'))['Audio_data'] 
        # Z-normalization
        data = (data-self.Feat_mean)/self.Feat_std
        data = data.reshape(-1)
        # Bounded NormFeat Range -3~3 and assign NaN to 0
        data[np.isnan(data)]=0
        data[data>3]=3
        data[data<-3]=-3        
        # Loading Label & Normalization
        label = self._labels[idx]
        label = (label-self.Label_mean)/self.Label_std
        return data, label

