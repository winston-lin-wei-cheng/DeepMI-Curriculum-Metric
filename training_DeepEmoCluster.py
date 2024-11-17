#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The code is adapted from: https://github.com/facebookresearch/deepcluster/blob/main/main.py

@author: winston
"""
import os
from tqdm import tqdm
from utils import cc_coef, AverageMeter, Logger, UnifLabelSampler 
from sklearn.metrics.cluster import normalized_mutual_info_score
import clustering
from dataloader import MspPodcastEmoDataset, UnlabelDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import numpy as np
import matplotlib.pyplot as plt
import models
import argparse



def compute_features(dataloader, model, N, dataset_type):
    if verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, batch_data in enumerate(dataloader):
        if dataset_type == 'supervised':
            input_tensor, target = batch_data
        elif dataset_type == 'non-supervised':
            input_tensor = batch_data
        
        input_data = torch.autograd.Variable(input_tensor.cuda())
        input_data = input_data.float()
        aux, _ = model(input_data)
        aux = aux.data.cpu().numpy()
        
        # create feature matrix
        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * batch_size: (i + 1) * batch_size] = aux
        else:
            # special treatment for final batch
            features[i * batch_size:] = aux

        # measure elapsed time
        torch.cuda.empty_cache()
        batch_time.update(time.time() - end)
        end = time.time()

    return features  


def train_class(loader, model, crit, opt, epoch):
    """ Self-Supervised of the ResNet model for acoustic clusters.
    
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): DenseResNet
            crit (torch.nn): loss of cluster classification (i.e., cross entropy)
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer           
            epoch (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.train()

    # freeze emotion regressor parameters
    for param in model.dnn_features.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.top_layer_class.parameters():
        param.requires_grad = True
    for param in model.emo_regressor.parameters():
        param.requires_grad = False
        
    # create an optimizer for the last fc layer (cluster-classification)
    optimizer_tl = torch.optim.SGD(model.top_layer_class.parameters(), lr=0.01)
    
    end = time.time()
    for i, (input_tensor, class_tar) in enumerate(tqdm(loader)):
        data_time.update(time.time() - end)
        # input data to GPU tensor
        input_data = torch.autograd.Variable(input_tensor.cuda())
        input_data = input_data.float()
        
        # input labels to GPU tensor
        input_class_tar = torch.autograd.Variable(class_tar.cuda())
        
        # model flow
        pred_class, _ = model(input_data)
        
        # loss calculation
        loss = crit(pred_class, input_class_tar)
        
        # record loss
        losses.update(loss.data.item(), input_tensor.size(0))
        
        # compute gradient and do optimizer step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()
        
        # measure elapsed time
        torch.cuda.empty_cache()
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg


def train_joint(loader, model, 
                crit_class, crit_attri,
                opt_class, opt_attri,
                epoch):
    """ Joint-training of the ResNet model for emotional clusters.
    
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): DenseResNet
            crit_class (torch.nn): loss of cluster classification (i.e., cross entropy)
            crit_attri (torch.nn): loss of attribute emotion regressions (i.e., CCC)
            opt_class (torch.optim.Adam): optimizer for every parameters with True
                                   requires_grad in model except top layer
            opt_attri (torch.optim.Adam): optimizer for every parameters with True
                                   requires_grad in model                                               
            epoch (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.train()

    # freeze emotion regressor parameters
    for param in model.dnn_features.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.top_layer_class.parameters():
        param.requires_grad = True
    for param in model.emo_regressor.parameters():
        param.requires_grad = True

    # create an optimizer for the last fc layer (cluster-classification)
    optimizer_tl = torch.optim.SGD(model.top_layer_class.parameters(), lr=0.01)
    
    end = time.time()
    for i, (input_tensor, class_tar, attri_tar) in enumerate(tqdm(loader)):
        data_time.update(time.time() - end)
        # input data to GPU tensor
        input_data = torch.autograd.Variable(input_tensor.cuda())
        input_data = input_data.float()
        
        # input labels to GPU tensor
        input_class_tar = torch.autograd.Variable(class_tar.cuda())
        attri_tar = attri_tar.view(attri_tar.size(0), 1)  # match shape for the CCC
                                                          # loss calculation (important notice!)
        input_attri_tar = torch.autograd.Variable(attri_tar.cuda())
        input_attri_tar = input_attri_tar.float()
        
        # model flow
        pred_class, pred_attri = model(input_data)
        
        # loss calculation
        loss_cluster = crit_class(pred_class, input_class_tar)
        loss_attri = crit_attri(pred_attri, input_attri_tar)
        loss = loss_cluster + loss_attri
        
        # record loss
        losses.update(loss.data.item(), input_tensor.size(0))
        
        # compute gradient and do optimizer step
        opt_class.zero_grad()
        opt_attri.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt_class.step()
        opt_attri.step()
        optimizer_tl.step()
            
        # measure elapsed time
        torch.cuda.empty_cache()
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg


def validation(loader, model, crit):
    """Validation of the ResNet model based on development set. (consider the supervised CCC loss only).
    
        Args:
            loader (torch.utils.data.DataLoader): validation data loader
            model (nn.Module): trained DenseNet
            crit (torch.nn): calculate validation/development loss           
    """
    # validation process
    model.eval()
    batch_loss_valid = []
    for i, (input_tensor, target) in enumerate(tqdm(loader)):    
        # Input Tensor Data
        input_var = torch.autograd.Variable(input_tensor.cuda())
        input_var = input_var.float()
        # Input Tensor Target
        target = target.view(target.size(0), 1)  # match shape for the CCC
                                                 # loss calculation (important notice!)
        target_var = torch.autograd.Variable(target.cuda())
        target_var = target_var.float()
        # models flow
        with torch.no_grad():
            _, pred_attri = model(input_var)
        # loss calculation
        loss = crit(pred_attri, target_var)
        batch_loss_valid.append(loss.data.cpu().numpy())
        torch.cuda.empty_cache()
    return np.mean(batch_loss_valid)
###############################################################################



argparse = argparse.ArgumentParser()
argparse.add_argument("-ep", "--epoch", required=True)
argparse.add_argument("-batch", "--batch_size", required=True)
argparse.add_argument("-emo", "--emo_attr", required=True)
argparse.add_argument("-nc", "--num_clusters", required=True)
args = vars(argparse.parse_args())

# Dirs & Parameters
root_dir = '/YOUR/PATH/TO/MSP-PODCAST-Publish-1.8/Features/OpenSmile_func_IS13ComParE/feat_mat/'
label_dir = '/YOUR/PATH/TO/MSP-PODCAST-Publish-1.8/Labels/labels_concensus.csv'
unlabel_dir = '/YOUR/PATH/TO/Some/Unlabeled_SpeechDataSet/Features/OpenSmile_func_IS13ComParE/feat_mat/'
batch_size = int(args['batch_size'])
epochs = int(args['epoch'])
emo_attr = args['emo_attr']
num_clusters = int(args['num_clusters'])
arch = 'resnet'
exp = './Models/'
reassign = 1
verbose = True

# fix random seeds (optional)
seed = 31
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)   

# loading the DenseResNet model
if verbose:
    print('Architecture: {}'.format(arch))
model = models.__dict__[arch](inp=6373, out=num_clusters)
fd = int(model.top_layer_class.weight.size()[1])
model.top_layer_class = None
model.dnn_features = torch.nn.DataParallel(model.dnn_features)
model.cuda()
cudnn.benchmark = True

# create optimizer
optimizer_class = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer_attri = torch.optim.Adam(model.parameters(), lr=0.0001)

# define the loss function for deepcluster classification
criterion_class = nn.CrossEntropyLoss().cuda()

# creating checkpoint repo
if not os.path.isdir(exp):
    os.makedirs(exp)

# creating cluster assignments log
cluster_log = Logger(os.path.join(exp, 'clusters'))

# emotional train/validation/unlabeled datasets
end = time.time()
unlabel_dataset = UnlabelDataset(unlabel_dir)
training_dataset = MspPodcastEmoDataset(root_dir, label_dir, split_set='Train', emo_attr=emo_attr)
validation_dataset = MspPodcastEmoDataset(root_dir, label_dir, split_set='Validation', emo_attr=emo_attr)

# shuffle training set by generating random indices 
valid_indices = list(range(len(validation_dataset)))
train_indices = list(range(len(training_dataset)))
unlabel_indices = list(range(len(unlabel_dataset)))

# creating data samplers and loaders
# NOTE: training loader cannot shuffle index !! (clustering would do this)
unlabel_loader = DataLoader(
    unlabel_dataset,
    batch_size=batch_size,
    num_workers=12,
    pin_memory=True,
)
train_loader = DataLoader(
    training_dataset,
    batch_size=batch_size,
    num_workers=12,
    pin_memory=True,
)
valid_sampler = SubsetRandomSampler(valid_indices)
valid_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    sampler=valid_sampler,
    num_workers=12,
    pin_memory=True,
)

if verbose:
    print('Load dataset: {0:.2f} s'.format(time.time() - end))

# clustering algorithm to use
deepcluster = clustering.__dict__['Kmeans'](num_clusters)

# Training DenseResNet with DeepEmoCluster
NMI = []
Loss_Joint = []
Loss_Class = []
Loss_CCC_Valid = []
loss_valid_best = 0
for epoch in range(epochs):
    end = time.time()
    # remove head
    model.top_layer_class = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    
    ###########################################################################
    # if apply semi-supervised learning (SSL) => Stage1: unsupervised part
    features_unlabel = compute_features(unlabel_loader, model, len(unlabel_dataset), dataset_type='non-supervised') 
    clustering_loss_unlabel = deepcluster.cluster(features_unlabel, verbose=verbose)
    cluster_training_dataset = clustering.cluster_assign(
        deepcluster.images_lists,
        unlabel_dataset.imgs,
        dataset_type='non-supervised',
    )
    unlabel_sampler = UnifLabelSampler(
        int(reassign * len(cluster_training_dataset)),
        deepcluster.images_lists,
    )  
    cluster_dataloader = DataLoader(
        cluster_training_dataset,
        batch_size=batch_size,
        num_workers=12,
        sampler=unlabel_sampler,
        pin_memory=True,
    )
    # add back head
    mlp = list(model.classifier.children())
    mlp.append(nn.ReLU(inplace=True).cuda())
    model.classifier = nn.Sequential(*mlp)
    model.top_layer_class = nn.Linear(fd, len(deepcluster.images_lists))
    model.top_layer_class.weight.data.normal_(0, 0.01)
    model.top_layer_class.bias.data.zero_()
    model.top_layer_class.cuda()
    loss_class_unlabel = train_class(cluster_dataloader, model, criterion_class, optimizer_class, epoch)
    Loss_Class.append(loss_class_unlabel)
    # remove head for next stage
    model.top_layer_class = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    ###########################################################################
    
    # get the DenseResNet features for the training set => Stage2: joint optimization part
    features = compute_features(train_loader, model, len(training_dataset), dataset_type='supervised')  

    # cluster the features
    if verbose:
        print('Cluster the features')
    clustering_loss = deepcluster.cluster(features, verbose=verbose)
    
    # assign pseudo-labels
    if verbose:
        print('Assign pseudo labels')
    emo_cluster_training_dataset = clustering.cluster_assign(
        deepcluster.images_lists,
        training_dataset.imgs,
        dataset_type='supervised',
    )    

    # uniformly sample per target
    sampler = UnifLabelSampler(
        int(reassign * len(emo_cluster_training_dataset)),
        deepcluster.images_lists,
    )

    emo_cluster_dataloader = DataLoader(
        emo_cluster_training_dataset,
        batch_size=batch_size,
        num_workers=12,
        sampler=sampler,
        pin_memory=True,
    )

    # set last fully connected layer
    mlp = list(model.classifier.children())
    mlp.append(nn.ReLU(inplace=True).cuda())
    model.classifier = nn.Sequential(*mlp)
    model.top_layer_class = nn.Linear(fd, len(deepcluster.images_lists))
    model.top_layer_class.weight.data.normal_(0, 0.01)
    model.top_layer_class.bias.data.zero_()
    model.top_layer_class.cuda()

    # Joint training for emotional clusters
    end = time.time()
    print('======== Epoch '+str(epoch)+' ========')       
    loss_joint = train_joint(
        emo_cluster_dataloader,
        model,
        criterion_class,
        cc_coef,
        optimizer_class,
        optimizer_attri,
        epoch,
    )   
    Loss_Joint.append(loss_joint)   
    print('Loss Joint: '+str(loss_joint))
    
    try:
        # monitor cluster reassignment results by NMI (optional)
        print('=====================================')
        nmi = normalized_mutual_info_score(
            clustering.arrange_clustering(deepcluster.images_lists),
            clustering.arrange_clustering(cluster_log.data[-1])
        )
        NMI.append(nmi)
        print('NMI against previous assignment: {0:.3f}'.format(nmi))
    except IndexError:
        pass
    
    # Validation Stage: considering the CCC performance only
    loss_valid = validation(valid_loader, model, cc_coef)
    Loss_CCC_Valid.append(loss_valid)   
    cluster_log.log(deepcluster.images_lists)
    print('Loss Validation CCC: '+str(loss_valid))
    
    # save model checkpoint based on the best validation CCC performance
    if epoch == 0:
        # initial CCC value
        loss_valid_best = loss_valid
        print("=> Saving the initial best model (Epoch="+str(epoch)+")")
        # save running checkpoint
        torch.save(
            {'arch': arch, 'state_dict': model.state_dict()},
            os.path.join(exp, 'ResNetDeepEmoCluster_epoch'+str(epochs)+'_batch'+str(batch_size)+'_'+str(num_clusters)+'clusters_'+emo_attr+'.pth.tar'),
        )
        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)
        
    else:     
        if loss_valid < loss_valid_best:
            print("=> Saving the best model (Epoch="+str(epoch)+")")
            print("=> CCC loss decrease from "+str(loss_valid_best)+" to "+str(loss_valid) )
            # save running checkpoint
            torch.save(
                {'arch': arch, 'state_dict': model.state_dict()},
                os.path.join(exp, 'ResNetDeepEmoCluster_epoch'+str(epochs)+'_batch'+str(batch_size)+'_'+str(num_clusters)+'clusters_'+emo_attr+'.pth.tar'),
            )
            # save cluster assignments
            cluster_log.log(deepcluster.images_lists) 
            # update best CCC value
            loss_valid_best = loss_valid
        else:
            print("=> CCC did not improve (Epoch="+str(epoch)+")")
    print('=================================================================')              
    
# save joint-train-loss/nmi/val-ccc trend for epochs
plt.plot(NMI,'bo--')
plt.plot(Loss_CCC_Valid,'ro--')
plt.plot(Loss_Joint,'ko--')
plt.savefig(exp + 'JointTrain-NMI-CCC_trend_epoch'+str(epochs)+'_batch'+str(batch_size)+'_'+str(num_clusters)+'clusters_'+emo_attr+'.png')

