
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
from torch.nn.functional import kl_div, softmax, log_softmax, one_hot
from torch import Tensor
from typing import Union
import torchvision



class KLDivLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(KLDivLoss, self).__init__()

        self.temperature = temperature
    def forward(self, emb1, emb2):
        emb1 = softmax(emb1/self.temperature, dim=1).detach()
        emb2 = log_softmax(emb2/self.temperature, dim=1)
        loss_kldiv = kl_div(emb2, emb1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        loss_kldiv = torch.mean(loss_kldiv)
        return loss_kldiv
 
class RankingLoss(nn.Module):
    def __init__(self, neg_penalty=0.03):
        super(RankingLoss, self).__init__()

        self.neg_penalty = neg_penalty
    def forward(self, ranks, labels, class_ids_loaded, device):
        '''
        for each correct it should be higher then the absence 
        '''
        labels = labels[:, class_ids_loaded]
        ranks_loaded = ranks[:, class_ids_loaded]
        neg_labels = 1+(labels*-1)
        loss_rank = torch.zeros(1).to(device)
        for i in range(len(labels)):
            correct = ranks_loaded[i, labels[i]==1]
            wrong = ranks_loaded[i, neg_labels[i]==1]
            correct = correct.reshape((-1, 1)).repeat((1, len(wrong)))
            wrong = wrong.repeat(len(correct)).reshape(len(correct), -1)
            image_level_penalty = ((self.neg_penalty+wrong) - correct)
            image_level_penalty[image_level_penalty<0]=0
            loss_rank += image_level_penalty.sum()
        loss_rank /=len(labels)

        return loss_rank
    
class CosineLoss(nn.Module):
    
    def forward(self, cxr, ehr ):
        a_norm = ehr / ehr.norm(dim=1)[:, None]
        b_norm = cxr / cxr.norm(dim=1)[:, None]
        loss = 1 - torch.mean(torch.diagonal(torch.mm(a_norm, b_norm.t()), 0))
        
        return loss



class FocalLoss(nn.Module):
    def __init__(self,gamma=2,alpha=0.25,reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        return torchvision.ops.focal_loss.sigmoid_focal_loss(pred, target,alpha=self.alpha,gamma=self.gamma,reduction=self.reduction)


class FocalLoss_learnable(nn.Module):
    def __init__(self,num_tasks=25,alpha=-1,init_gamma=2,reduction='mean',device='cpu'):
        super().__init__()
        self.alpha = alpha
        self.log_gamma = torch.nn.Parameter(torch.zeros(num_tasks, requires_grad=True, dtype=torch.float32, device=device)) 
        self.reduction = reduction

    def forward(self, pred, target):
        self.gamma = torch.exp(self.log_gamma)
        return torchvision.ops.focal_loss.sigmoid_focal_loss(pred, target,alpha=self.alpha,gamma=self.gamma,reduction=self.reduction) + 0.5 * torch.mean(self.log_gamma)


class MultiLoss(torch.nn.Module):
    def __init__(self, num_tasks=25, device='cpu'):
        super().__init__()
        self.log_var = torch.nn.Parameter(torch.zeros(num_tasks, requires_grad=True, dtype=torch.float32, device=device)) 

    def forward(self, pred, target):
        self.sigmas_sq = torch.exp(-self.log_var)
        losses = F.binary_cross_entropy(pred, target, reduction='none')
        losses = torch.mean(losses, axis=0)
        loss = losses * self.sigmas_sq + 0.5 * self.log_var
        loss = torch.mean(loss)
        return loss
