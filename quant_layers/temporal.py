import os

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np 
import tqdm
import math
from functools import partial

class Temporal_Module(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = int(os.environ.get('TIME_EMBED_DIM'))
        dropout_ratio = float(os.environ.get('DROPOUT', 0.2))
        
        hidden_dim = 64
        self.s_gen = nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout_ratio),
                            nn.Linear(hidden_dim, 1),
                            nn.Softplus()
                    )
    
    def initialize(self, bb):
        self.s_gen[-2].bias.data.fill_(torch.log(torch.exp(bb) - 1.))

    def forward(self, time_emb):
        return self.s_gen(time_emb)

class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):    
        return input.round()
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_lv, size):
        ctx.save_for_backward(torch.Tensor([n_lv, size]))
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        saved, = ctx.saved_tensors
        n_lv, size = int(saved[0]), float(saved[1])

        if n_lv == 0:
            return grad_output, None, None
        else:
            scale = 1 / np.sqrt(n_lv * size)
            return grad_output.mul(scale), None, None