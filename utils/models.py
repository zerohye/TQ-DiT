from types import MethodType
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models import vision_transformer
from timm.models.vision_transformer import Attention, Mlp
from timm.models.swin_transformer import WindowAttention
from utils.nets import DiT_models
from download import find_model
from quant_layers.matmul import SoSPTQSLBatchingQuantMatMul, TimewiseSoSPTQSLBatchingQuantMatMul

from utils.global_var import get_model_input, hook

def attention_forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

    # attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    del q, k

    # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    
    from utils.global_var import get_model_input, hook
    model_input = get_model_input()
    t_ind = 0
    if attn[t_ind] != []:
        t = model_input[t_ind]
        if t % 10 == 0:
            if hasattr(t, 'cpu'):
                hook(t.cpu(), attn.cpu(), t_ind)
            else:
                hook(t, attn.cpu(), t_ind)
    
    SoS_hook(self,attn)    
    
    if isinstance(self.matmul2, TimewiseSoSPTQSLBatchingQuantMatMul):
        timegroup_ind = get_model_input()
        x = self.matmul2(attn, v, timegroup_ind = timegroup_ind).transpose(1, 2).reshape(B, N, C)
    else:
        x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    del attn, v
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def SoS_hook(self, x):
    from utils.global_var import get_model_input, hook
    model_input_t = get_model_input()
    plot = False
    timegroup = False
    
    if plot:
        if model_input_t is not None:
            t_ind = 0
            if x[t_ind] != []:
                t = model_input_t[t_ind][0]
                # if t % 10 == 0:
                hook(t.cpu().detach(), x.cpu().detach(), t_ind)
                
    if timegroup:
        if model_input_t is not None:
            calib_size = 8
            module = self.matmul2
            total_timestep = 250
            timegroup_num = 25
            timestepnum_per_timegroup = total_timestep // timegroup_num
            
            for t_ind in range(calib_size):
                t = model_input_t[t_ind]
                timegroup_ind = min(t // timestepnum_per_timegroup, total_timestep//timestepnum_per_timegroup-1)     
                if not hasattr(self.matmul2, 'timestep'):
                    module.timestep = []
                module.timestep.append(timegroup_ind.cpu().detach())

def window_attention_forward(self, x, mask = None):
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    q = q * self.scale
    # attn = (q @ k.transpose(-2, -1))
    attn = self.matmul1(q, k.transpose(-2,-1))

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B

# def mlp_forward(self, x):
    
#     x = self.fc1(x)
#     x = self.act(x)
#     x = self.drop(x)    
#     # Gelu_hook(self, x)        
#     x = self.fc2(x)
#     x = self.drop(x)
#     return x

def Gelu_hook(self, x):
    from utils.global_var import get_model_input, hook
    model_input_t = get_model_input()
    plot = False
    timegroup = True
    
    if plot:
        if model_input_t is not None:
            t_ind = 0
            if x[t_ind] != []:
                t = model_input_t[t_ind][0]
                # if t % 10 == 0:
                hook(t.cpu().detach(), x.cpu().detach(), t_ind)
                
    if timegroup:
        module = self.fc2
        t = model_input_t[t_ind][0]
        total_timestep = 250
        timegroup_num = 25
        timestepnum_per_timegroup = total_timestep // timegroup_num
        timegroup_ind = min(t // timestepnum_per_timegroup, total_timestep//timestepnum_per_timegroup-1)     
        
        if module.timestep is None:
            module.timestep = []
        module.timestep.append(timegroup_ind.cpu().detach())

    
def get_net(name, image_size, num_classes, device, ckpt):
    
    latent_size = image_size // 8
    net = DiT_models[name](
        input_size=latent_size,
        num_classes=num_classes
    ).to(device)
    ckpt_path = ckpt or f"DiT-XL-2-{image_size}x{image_size}.pt"
    state_dict = find_model(ckpt_path)
    net.load_state_dict(state_dict)
    net.eval()  # important!

    for name, module in net.named_modules():
        if isinstance(module, Attention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(attention_forward, module)
        if isinstance(module, WindowAttention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(window_attention_forward, module)
        # if isinstance(module, Mlp):
        #     module.forward = MethodType(mlp_forward, module)

    net.to(device)
    net.eval()
    return net
