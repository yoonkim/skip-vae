#!/usr/bin/env python3
import sys
import os

import argparse
import json
import random
import shutil

import torch
from torch import cuda
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

def log_bernoulli_loss(pred, y, min_eps=1e-5, max_eps=1.-1e-5, average=True):
  prob = torch.clamp(pred.view(pred.size(0), -1), min=min_eps, max = max_eps)
  y_vec = y.view(y.size(0), -1)
  log_bernoulli = y_vec * torch.log(prob) + (1. - y_vec)*torch.log(1. - prob)
  if average:
    return -torch.mean(torch.sum(log_bernoulli, 1))
  else: 
    return -torch.sum(log_bernoulli, 1)

def logsumexp(x, dim=1):
  d = torch.max(x, dim)[0]    
  if x.dim() == 1:
    return torch.log(torch.exp(x - d).sum(dim)) + d
  else:
    return torch.log(torch.exp(x - d.unsqueeze(dim).expand_as(x)).sum(dim)) + d
  
# def logsumexp(x):
#   max_x = torch.max(x, 1)[0]
#   new_x = x - max_x.unsqueeze(1).expand_as(x)
#   return max_x + (new_x.exp().sum(1)).log()

def kl_loss(mean1, logvar1, mean2, logvar2):
  result =  -0.5 * torch.sum(logvar1 - logvar2 - torch.pow(mean1-mean2, 2) / logvar2.exp() -
                            torch.exp(logvar1 - logvar2) + 1, 1)
  return result.mean()

def kl_loss_dim(mean1, logvar1):
  result =  -0.5 * (logvar1 - torch.pow(mean1, 2)- torch.exp(logvar1) + 1)
  return result

def kl_loss_diag(mean, logvar, logvar_prior=None, average=True):
  if logvar_prior is None:
    result = -0.5 * torch.sum(logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1, 1)
  else:
    logvar_prior = logvar_prior.unsqueeze(0).expand_as(logvar)        
    result =  -0.5 * torch.sum(logvar - logvar_prior - torch.pow(mean, 2) / logvar_prior.exp() -
                            torch.exp(logvar - logvar_prior) + 1, 1)
  if average:
    return result.mean()
  else:
    return result

def log_gaussian(z, mean, logvar):
  d = mean.size(1)
  k = mean.size(0)
  var = logvar.exp()
  log_density = ((z - mean)**2 / var).sum(1) # b
  log_det = logvar.sum(1) # b
  log_density = log_density + log_det + d*np.log(2*np.pi)
  return -0.5*log_density
  
def is_weight(z, mean, logvar):
  # returns log p(z) / q(z | x)
  var = logvar.exp()  
  q_term = ((z - mean)**2 / var).sum(1)
  p_term = (z ** 2).sum(1)
  combined = 0.5 * (q_term - p_term)  
  det = 0.5*(logvar.sum(1))
  return det + combined
  
