import random
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import torch 
import torch.backends.cudnn 
import torch.cuda

def set_determenistic_mode(SEED, disable_cudnn):
  torch.manual_seed(SEED)
  random.seed(SEED)                          
  rs = RandomState(MT19937(SeedSequence(SEED))) 
  np.random.seed(SEED)             
  torch.cuda.manual_seed_all(SEED)             

  if not disable_cudnn:
    torch.backends.cudnn.benchmark = False    
    #torch.backends.cudnn.deterministic = True 

  else:
    torch.backends.cudnn.enabled = False 
