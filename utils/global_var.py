from collections import defaultdict
import numpy as np
import torch
model_input = None
hook_data = defaultdict(list)
flop = 0

def set_model_input(x):
    global model_input
    model_input = x

def get_model_input():
    global model_input
    return model_input

def hook(t, input, t_ind):
    global hook_data
    key = "{}".format(t)
    if hook_data.get(key) is None:
        hook_data[key].append(input[t_ind])
    
def get_hook_data():
    global hook_data
    return hook_data

def get_group_ind(t, option = "calib", total_timestep = 250, group_num = 25):
    group_interval = total_timestep // group_num # base 10
    if option == "calib":
        ind = min(t[0] // (group_interval*4), total_timestep//group_interval-1).cpu().detach().item()
    elif option == "inference":
        ind = []
        for i in range(len(t)):
            # ind.append(min(t[i] // (group_interval*4), total_timestep//group_interval-1).cpu().detach().item()) #이전, total_timestep과 group_interval 고정
            if hasattr(min(t[i] // group_interval, group_num-1), 'cpu'):
                ind.append(min(t[i] // group_interval, group_num-1).cpu().detach().item()) #이후
            else:
                ind.append(min(t[i] // group_interval, group_num-1)) #이후
    return ind

def FLOP(flop_ = None):
    global flop
    if flop_:
        flop = flop_
    else:
        return flop