import numpy as np
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
from itertools import product     
import tqdm
from YH_weight_distribution_plot import weight_plot, activation_box_plot
from utils.global_var import get_group_ind, get_model_input

class MinMaxQuantMatMul(nn.Module):
    """Matrix Multiplication base class"""
    def __init__(self, A_bit=8, B_bit=8, mode="raw", sym=False):
        super().__init__()
        self.A_bit=A_bit
        self.B_bit=B_bit
        self.A_interval=None
        self.B_interval=None
        self.A_qmax=2**(self.A_bit-1) # 8bit(256) -> -127 ~ 128
        self.B_qmax=2**(self.B_bit-1)
        self.mode=mode
        self.raw_input = None
        self.raw_out = None
        self.sym = sym
    
    def forward(self, A,B):
        if self.mode=='raw':
            out=A @ B
        elif self.mode=="quant_forward":
            out=self.quant_forward(A,B)
        elif self.mode=="calibration_step1":
            out=self.calibration_step1(A,B)
        elif self.mode=="calibration_step2":
            out=self.calibration_step2(A,B)
        else:
            raise NotImplementedError
        return out
    
    def quant_input(self,x,interval,qmax):
        x_sim=(x/interval).round_().clamp_(-qmax,qmax-1)
        x_sim.mul_(interval)
        return x_sim
    
    def quant_forward(self,A,B):
        # assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        assert self.A_interval is not None,f"You should define self.A_interval before run quant_forward for {self}"
        A_sim=self.quant_input(A,self.A_interval,self.A_qmax)
        B_sim=self.quant_input(B,self.B_interval,self.B_qmax)
        out=A_sim@B_sim
        return out

    def calibration_step1(self,A,B):
        # step1: collection the FP32 values
        self.raw_input=A.cpu().detach(), B.cpu().detach()
        out=A@B
        self.raw_out=out.cpu().detach()
        return out
    
    def calibration_step2(self,A,B):
        # step2: search for the best S^w and S^o of each layer
        self.A_interval=(A.data.abs().max()/(self.A_qmax-0.5)).detach()
        self.B_interval=(B.data.abs().max()/(self.B_qmax-0.5)).detach()
        self.calibrated=True
        out=self.quant_forward(A,B)        
        return out

class PTQSLQuantMatMul(MinMaxQuantMatMul):
    """
    Chunk matrix into blockes and quantize.
    Chunking follows naive padding strategy.
    Alternately search for best intervals of each individual blocks for A and B.

    two different scenarios:
    - Q @ K:
        - A's shape: B,H,S,W
        - B's shape: B,H,W,S
    - scores @ V:
        - A's shape: B,H,S,S
        - B's shape: B,H,S,W
    - interval shape: 1,n_G,1,n_V,1,n_H,1
    """
    def __init__(self, A_bit=8, B_bit=8, sym=False, mode="raw",
                 metric="L2_norm", search_round=1, eq_alpha=0.1, eq_beta=2, eq_n=100, parallel_eq_n=10,
                 n_G_A=1, n_V_A=1, n_H_A=1, n_G_B=1, n_V_B=1, n_H_B=1, init_layerwise=False):
        super().__init__(A_bit=A_bit, B_bit=B_bit, mode=mode, sym=sym)
        self.metric = metric
        self.search_round = search_round
        self.eq_alpha = eq_alpha
        self.eq_beta = eq_beta
        self.eq_n = eq_n
        self.parallel_eq_n = parallel_eq_n
        self.n_G_A = n_G_A
        self.n_V_A = n_V_A
        self.n_H_A = n_H_A
        self.n_G_B = n_G_B
        self.n_V_B = n_V_B
        self.n_H_B = n_H_B
        # init these parameters in self.calibration_step1
        self.crb_groups_A = None
        self.crb_groups_B = None
        self.crb_rows_A = None
        self.crb_cols_A = None
        self.crb_rows_B = None
        self.crb_cols_B = None
        self.pad_groups_A = None
        self.pad_groups_B = None
        self.pad_rows_A = None
        self.pad_rows_B = None
        self.pad_cols_A = None
        self.pad_cols_B = None
        self.raw_grad = None
        self.init_layerwise = init_layerwise
        self.sym = sym

    def _get_padding_parameters_with_shape(self):
        assert self.A_shape is not None,f"You should define self.A_shape before run quant_forward for {self}"
        A_shape = self.A_shape
        B_shape = self.B_shape
        self.crb_groups_A = (A_shape[1]+self.n_G_A-1) // self.n_G_A
        self.crb_groups_B = (B_shape[1]+self.n_G_B-1) // self.n_G_B
        self.crb_rows_A = (A_shape[2]+self.n_V_A-1) // self.n_V_A
        self.crb_cols_A = (A_shape[3]+self.n_H_A-1) // self.n_H_A
        self.crb_rows_B = (B_shape[2]+self.n_V_B-1) // self.n_V_B
        self.crb_cols_B = (B_shape[3]+self.n_H_B-1) // self.n_H_B

        self.pad_groups_A = self.crb_groups_A*self.n_G_A - A_shape[1]
        self.pad_rows_A = self.crb_rows_A*self.n_V_A - A_shape[2]
        self.pad_cols_A = self.crb_cols_A*self.n_H_A - A_shape[3]
        self.pad_groups_B = self.crb_groups_B*self.n_G_B - B_shape[1]
        self.pad_rows_B = self.crb_rows_B*self.n_V_B - B_shape[2]
        self.pad_cols_B = self.crb_cols_B*self.n_H_B - B_shape[3]
        
    def _get_padding_parameters(self, A, B):
        self.A_shape = A.shape
        self.B_shape = B.shape
        self.crb_groups_A = (A.shape[1]+self.n_G_A-1) // self.n_G_A
        self.crb_groups_B = (B.shape[1]+self.n_G_B-1) // self.n_G_B
        self.crb_rows_A = (A.shape[2]+self.n_V_A-1) // self.n_V_A
        self.crb_cols_A = (A.shape[3]+self.n_H_A-1) // self.n_H_A
        self.crb_rows_B = (B.shape[2]+self.n_V_B-1) // self.n_V_B
        self.crb_cols_B = (B.shape[3]+self.n_H_B-1) // self.n_H_B

        self.pad_groups_A = self.crb_groups_A*self.n_G_A - A.shape[1]
        self.pad_rows_A = self.crb_rows_A*self.n_V_A - A.shape[2]
        self.pad_cols_A = self.crb_cols_A*self.n_H_A - A.shape[3]
        self.pad_groups_B = self.crb_groups_B*self.n_G_B - B.shape[1]
        self.pad_rows_B = self.crb_rows_B*self.n_V_B - B.shape[2]
        self.pad_cols_B = self.crb_cols_B*self.n_H_B - B.shape[3]

    def quant_input_A(self, x):
        A_interval = self.A_interval.reshape(1, self.A_interval.shape[1], 1, 1)
        x = (x/A_interval).round_().clamp(-self.A_qmax,self.A_qmax-1).mul_(A_interval)
        
        
        # x = F.pad(x, [0,self.pad_cols_A,0,self.pad_rows_A,0,self.pad_groups_A])
        # x = x.view(-1,self.n_G_A,self.crb_groups_A,self.n_V_A,self.crb_rows_A,self.n_H_A,self.crb_cols_A)
        # x = (x/self.A_interval).round_().clamp(-self.A_qmax,self.A_qmax-1).mul_(self.A_interval)
        # x = x.view(-1,self.n_G_A*self.crb_groups_A,self.n_V_A*self.crb_rows_A,self.n_H_A*self.crb_cols_A)
        # x = x[:,:x.shape[1]-self.pad_groups_A,:x.shape[2]-self.pad_rows_A,:x.shape[3]-self.pad_cols_A]
        return x
    
    def quant_input_B(self, x):
        B_interval = self.B_interval.reshape(1, self.B_interval.shape[1], 1, 1)
        x = (x/B_interval).round_().clamp(-self.B_qmax,self.B_qmax-1).mul_(B_interval)
        
        # x = F.pad(x, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B])
        # x = x.view(-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)
        # x = (x/self.B_interval).round_().clamp(-self.B_qmax,self.B_qmax-1).mul_(self.B_interval)
        # x = x.view(-1,self.n_G_B*self.crb_groups_B,self.n_V_B*self.crb_rows_B,self.n_H_B*self.crb_cols_B)
        # x = x[:,:x.shape[1]-self.pad_groups_B,:x.shape[2]-self.pad_rows_B,:x.shape[3]-self.pad_cols_B]
        return x

    def quant_forward(self, A, B):
        # assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        assert self.A_interval is not None,f"You should define self.A_interval before run quant_forward for {self}"
        A_sim=self.quant_input_A(A)
        B_sim=self.quant_input_B(B)
        out=A_sim@B_sim
        return out

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None, dim=-1):
        """
        tensor_raw: *, features, *
        tensor_sim: *, features, *
        similarity: *
        It's your job to calculate mean on non-feature * dims!

        Similarity without inherent feature structure is more welcome to parallelism.
        """
        if metric == "cosine":
            similarity = F.cosine_similarity(tensor_raw, tensor_sim, dim=dim) # should only support dim=-1 and cannot be paralleled
        elif metric == "pearson":
            similarity = F.cosine_similarity(tensor_raw-torch.mean(tensor_raw), tensor_sim-torch.mean(tensor_sim), dim=dim)
        else:
            if metric == "L1_norm":
                similarity = -torch.abs(tensor_raw - tensor_sim)
            elif metric == "L2_norm":
                similarity = -(tensor_raw - tensor_sim) ** 2
            elif metric == "linear_weighted_L2_norm":
                similarity = -tensor_raw.abs() * (tensor_raw - tensor_sim) ** 2
            elif metric == "square_weighted_L2_norm":
                similarity = -(tensor_raw * (tensor_raw - tensor_sim)) ** 2
            elif metric == "hessian":
                raw_grad = self.raw_grad.reshape_as(tensor_raw)
                similarity = -(raw_grad * (tensor_raw - tensor_sim)) ** 2
            else:
                raise NotImplementedError(f"metric {metric} not implemented!")
            similarity = torch.mean(similarity, dim=dim)
        return similarity

    def _search_best_A_interval(self, A, B, A_interval_candidates):
        """
        Modularization of searching best interval
        """
        # recalculate A_pad
        A_pad = F.pad(A, [0,self.pad_cols_A,0,self.pad_rows_A,0,self.pad_groups_A]).unsqueeze(0).view(1,-1,self.n_G_A,self.crb_groups_A,self.n_V_A,self.crb_rows_A,self.n_H_A,self.crb_cols_A)

        tmp_A_interval = self.A_interval.unsqueeze(0) # shape: 1,1,n_G,1,n_V,1,n_H,1
        # out-of-loop optimization
        B_sim = self.quant_input_B(B).unsqueeze(0) # shape: 1,B,H,dim2,dim3
        for v, h in product(range(self.n_V_A), range(self.n_H_A)):
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n,p_st+self.parallel_eq_n)
                # quantize A
                cur_A_interval = tmp_A_interval.repeat(p_ed-p_st,1,1,1,1,1,1,1)
                cur_A_interval[:,:,:,:,v:v+1,:,h:h+1,:] = A_interval_candidates[p_st:p_ed,:,:,:,v:v+1,:,h:h+1,:]
                A_sim = (A_pad/cur_A_interval).round_().clamp_(-self.A_qmax,self.A_qmax-1).mul_(cur_A_interval)
                A_sim = A_sim.view(p_ed-p_st,-1,A.shape[1]+self.pad_groups_A,A.shape[2]+self.pad_rows_A,A.shape[3]+self.pad_cols_A) # shape: parallel_eq_n,B,H*,dim1*,dim2* (* stand for padding)
                A_sim = A_sim[:,:,:A.shape[1],:A.shape[2],:A.shape[3]] # shape: parallel_eq_n,B,H,dim1,dim2
                # quantize B, this quantization is optimized out of loop
                # calculate similarity and store them
                out_sim = A_sim @ B_sim # shape: parallel_eq_n,B,H,dim1,dim3
                similarity = self._get_similarity(self.raw_out, out_sim, self.metric) # shape: parallel_eq_n,B,H,dim1
                similarity = similarity.mean([1,3]) # shape: parallel_eq_n,H (remaining mean operation will be done later on)
                similarities.append(similarity)
            # calculate best similarity for this block
            similarities = torch.cat(similarities, 0) # shape: eq_n,H
            similarities = F.pad(similarities, [0,self.pad_groups_A]).view(self.eq_n,self.n_G_A,self.crb_groups_A).mean(-1) # shape: eq_n, n_G_A
            best_index = torch.argmax(similarities, dim=0, keepdim=False).view(1,1,-1,1,1,1,1,1)
            tmp_A_interval[:,:,:,:,v:v+1,:,h:h+1,:] = torch.gather(A_interval_candidates[:,:,:,:,v:v+1,:,h:h+1,:],dim=0,index=best_index)
        self.A_interval = tmp_A_interval.squeeze(0)

    def _search_best_B_interval(self, A, B, B_interval_candidates):
        """
        Modularization of searching best interval
        """
        # recalculate B_pad
        B_pad = F.pad(B, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B]).unsqueeze(0).view(1,-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)

        tmp_B_interval = self.B_interval.unsqueeze(0) # shape: 1,1,n_G,1,n_V,1,n_H,1
        # out-of-loop optimization
        A_sim = self.quant_input_A(A).unsqueeze(0) # shape: 1,B,H,dim1,dim2
        for v, h in product(range(self.n_V_B), range(self.n_H_B)):
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n,p_st+self.parallel_eq_n)
                # quantize A, this quantization is optimized out of loop
                # quantize B
                cur_B_interval = tmp_B_interval.repeat(p_ed-p_st,1,1,1,1,1,1,1)
                cur_B_interval[:,:,:,:,v:v+1,:,h:h+1,:] = B_interval_candidates[p_st:p_ed,:,:,:,v:v+1,:,h:h+1,:]
                B_sim = (B_pad/cur_B_interval).round_().clamp_(-self.B_qmax,self.B_qmax-1).mul_(cur_B_interval)
                B_sim = B_sim.view(p_ed-p_st,-1,B.shape[1]+self.pad_groups_B,B.shape[2]+self.pad_rows_B,B.shape[3]+self.pad_cols_B) # shape: parallel_eq_n,B,H*,dim2*,dim3* (* stand for padding)
                B_sim = B_sim[:,:,:B.shape[1],:B.shape[2],:B.shape[3]] # shape: parallel_eq_n,B,H,dim2,dim3
                # calculate similarity and store them
                out_sim = A_sim @ B_sim # shape: parallel_eq_n,B,H,dim1,dim3
                similarity = self._get_similarity(self.raw_out, out_sim, self.metric) # shape: parallel_eq_n,B,H,dim1
                similarity = similarity.mean([1,3]) # shape: parallel_eq_n,H (remaining mean operation will be done later on)
                similarities.append(similarity)
            # calculate best similarity for this block
            similarities = torch.cat(similarities, 0) # shape: eq_n,H
            similarities = F.pad(similarities, [0,self.pad_groups_B]).view(self.eq_n,self.n_G_B,self.crb_groups_B).mean(-1) # shape: eq_n, n_G_B
            best_index = torch.argmax(similarities, dim=0, keepdim=False).view(1,1,-1,1,1,1,1,1)
            tmp_B_interval[:,:,:,:,v:v+1,:,h:h+1,:] = torch.gather(B_interval_candidates[:,:,:,:,v:v+1,:,h:h+1,:],dim=0,index=best_index)
        self.B_interval = tmp_B_interval.squeeze(0)

    def _initialize_intervals(self, A, B):
        # pad A and B for future quantization
        self._get_padding_parameters(A, B) # put it here because hessian does not use calibration step 1
        A_pad = F.pad(A, [0,self.pad_cols_A,0,self.pad_rows_A,0,self.pad_groups_A]).unsqueeze(0).view(1,-1,self.n_G_A,self.crb_groups_A,self.n_V_A,self.crb_rows_A,self.n_H_A,self.crb_cols_A) # shape: 1,B,n_G,crb_groups,n_V,crb_rows,n_H,crb_cols
        B_pad = F.pad(B, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B]).unsqueeze(0).view(1,-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)

        # initialize intervals with minmax intervals
        if self.init_layerwise:
            self.A_interval = (A.abs().max()/(self.A_qmax-0.5)).detach().view(1,1,1,1,1,1,1).repeat(1,self.n_G_A,1,self.n_V_A,1,self.n_H_A,1)
            self.B_interval = (B.abs().max()/(self.B_qmax-0.5)).detach().view(1,1,1,1,1,1,1).repeat(1,self.n_G_B,1,self.n_V_B,1,self.n_H_B,1)
        else:
            self.A_interval=(A_pad.abs().amax([0,1,3,5,7], keepdim=True)/(self.A_qmax-0.5)).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1
            self.B_interval=(B_pad.abs().amax([0,1,3,5,7], keepdim=True)/(self.B_qmax-0.5)).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1

    def calibration_step2(self, A, B):
        # put raw outs/grads on GPU
        self.raw_out = self.raw_out.unsqueeze(0).to(A.device)
        self.raw_grad = self.raw_grad.to(A.device) if self.raw_grad != None else None

        self._initialize_intervals(A, B)

        # prepare weight intervals and similarities
        A_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(-1,1,1,1,1,1,1,1) * self.A_interval.unsqueeze(0)
        B_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(-1,1,1,1,1,1,1,1) * self.B_interval.unsqueeze(0)

        for e in range(self.search_round):
            # search for best A interval
            self._search_best_A_interval(A, B, A_interval_candidates)
            # search for best B interval
            self._search_best_B_interval(A, B, B_interval_candidates)

        # put raw data back to cpu
        self.raw_out = self.raw_out.squeeze(0).to("cpu")
        self.raw_grad = self.raw_grad.to("cpu") if self.raw_grad != None else None

        # finish calibration and output the result
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad
        out=self.quant_forward(A,B)
        return out    

class SoSPTQSLQuantMatMul(PTQSLQuantMatMul):
    """
    Sublayerwise PTQ on matmul modules with Split-of-Softmax (SoS) on score matrix.
    
    Data after softmaxing has highly biased distribution, making it difficult to quantize with uniform quantization.
    An elegant tradeoff between great majority of unimportant values and few crucial values is impossible under low bit quantization.
    Therefore, we propose to split complete interval of (0, 1) into several smaller intervals and perform uniform quantization on each.
    We could manually assgin or search for the best split point.
    Currently, we only consider single split point scenarios, since this proves to be effective enough.

    The algorithm no longer requires PTQSL on score matrix, and will ignore relevant parameters.

    with proper hardware implementation, we don't need to use a sign bit anymore.
    """
    def __init__(self, A_bit=8, B_bit=8, mode="raw",
                 metric="L2_norm", search_round=1, eq_alpha=0.1, eq_beta=2, eq_n=100, parallel_eq_n=10,
                 n_G_A=1, n_V_A=1, n_H_A=1, n_G_B=1, n_V_B=1, n_H_B=1, init_layerwise=False,
                 split=None):
        super().__init__(A_bit=A_bit, B_bit=B_bit, mode=mode, 
                         metric=metric, search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=parallel_eq_n, 
                         n_G_A=n_G_A, n_V_A=n_V_A, n_H_A=n_H_A, n_G_B=n_G_B, n_V_B=n_V_B, n_H_B=n_H_B, init_layerwise=init_layerwise)
        self.n_G_A = 1
        self.n_V_A = 1
        self.n_H_A = 1
        self.A_qmax = 2**(self.A_bit-1) # well, still need it 
        self.split = split
        if split != None:
            self.A_interval = self.split/(self.A_qmax-1)

    def quant_input_A(self, x):
        x_high = (x.clamp(self.split, 1)*(self.A_qmax-1)).round_().clamp_(0,self.A_qmax-1)/(self.A_qmax-1)
        x_low = (x.clamp(0, self.split)/self.A_interval).round_().clamp_(0,self.A_qmax-1)*self.A_interval
        return x_high + x_low

    def _search_best_A_interval(self, A, B, split_candidates):
        """
        search for best split point
        """
        # out-of-loop optimization
        A_ = A.unsqueeze(0)
        # B_sim = self.quant_input_B(B).unsqueeze(0) # shape: 1,B,H,dim2,dim3
        B_sim = B.unsqueeze(0)

        similarities = []
        for i in range(len(split_candidates)):
            # quantize A
            cur_A_interval = split_candidates[i]/(self.A_qmax-1)
            A_high = (A_.clamp(split_candidates[i], 1)*(self.A_qmax-1)).round_().clamp_(0,self.A_qmax-1)/(self.A_qmax-1)
            A_low =( A_.clamp(0, split_candidates[i])/cur_A_interval).round_().clamp_(0,self.A_qmax-1)*cur_A_interval
            A_sim = A_high + A_low # shape: 1,B,H,S,S
            # quantize B, this quantization is optimized out of loop
            # calculate similarity and store them (dim1=dim2=S, dim3=W)
            out_sim = A_sim @ B_sim # shape: 1,B,H,dim1,dim3
            similarity = self._get_similarity(self.raw_out, out_sim, self.metric) # shape: parallel_eq_n,B,H,dim1
            similarity = similarity.mean([1,2,3]) # shape: 1
            similarities.append(similarity)
        # calculate best similarity for this block
        similarities = torch.cat(similarities, 0) # shape: eq_n
        best_index = torch.argmax(similarities, dim=0, keepdim=False)
        self.split = split_candidates[best_index]
        self.A_interval = self.split/(self.A_qmax-1)
        # debugging
        # print(f"best split: {self.split}")

    def _initialize_intervals(self, A, B):
        # pad A and B for future quantization
        self._get_padding_parameters(A, B)
        B_pad = F.pad(B, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B]).unsqueeze(0).view(1,-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)

        # initialize intervals with minmax intervals
        self.split = 0.01
        self.A_interval = self.split/(self.A_qmax-1)
        if self.init_layerwise:
            self.B_interval = (B.abs().max()/(self.B_qmax-0.5)).detach().view(1,1,1,1,1,1,1).repeat(1,self.n_G_B,1,self.n_V_B,1,self.n_H_B,1)
        else:
            self.B_interval=(B_pad.abs().amax([0,1,3,5,7], keepdim=True)/(self.B_qmax-0.5)).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1
    
    def calibration_step2(self, A, B):
        # put raw outs/grads on GPU
        self.raw_out = self.raw_out.unsqueeze(0).to(A.device)
        self.raw_grad = self.raw_grad.to(A.device) if self.raw_grad != None else None

        self._initialize_intervals(A, B)

        # prepare weight intervals and similarities
        A_split_candidates = torch.tensor([2**(-i) for i in range(20)]).cuda()
        # split_eq_alpha, split_eq_beta, split_eq_n = 0.002, 0.03, 50
        # A_split_candidates = torch.tensor([split_eq_alpha + (split_eq_beta- split_eq_alpha)*i/split_eq_n for i in range(split_eq_n + 1)]).cuda()
        B_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(-1,1,1,1,1,1,1,1) * self.B_interval.unsqueeze(0)

        for e in range(self.search_round):
            # search for best A interval
            self._search_best_A_interval(A, B, A_split_candidates)
            # search for best B interval
            self._search_best_B_interval(A, B, B_interval_candidates)

        # put raw data back to cpu
        self.raw_out = self.raw_out.squeeze(0).to("cpu")
        self.raw_grad = self.raw_grad.to("cpu") if self.raw_grad != None else None

        # finish calibration and output the result
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad
        out=self.quant_forward(A,B)
        return out    

class PTQSLBatchingQuantMatMul(PTQSLQuantMatMul):
    def __init__(self, A_bit=8, B_bit=8, mode="raw", sym = False,
                 metric="L2_norm", search_round=1, eq_alpha=0.1, eq_beta=2, eq_n=100, parallel_eq_n=10,
                 n_G_A=1, n_V_A=1, n_H_A=1, n_G_B=1, n_V_B=1, n_H_B=1, init_layerwise=False, revised = False):
        self.device = 'cuda'
        self.sym = sym
        self.revised = revised
        super().__init__(A_bit=A_bit, B_bit=B_bit, sym = sym, mode=mode, metric=metric, search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=parallel_eq_n, n_G_A=n_G_A, n_V_A=n_V_A, n_H_A=n_H_A, n_G_B=n_G_B, n_V_B=n_V_B, n_H_B=n_H_B, init_layerwise=init_layerwise)

    def _initialize_calib_parameters(self):
        """ 
        set parameters for feeding calibration data
        """
        if isinstance(self.raw_input[0], list):
            self.calib_size = len(self.raw_input[0])*int(self.raw_input[0][0].shape[0])
            self.calib_batch_size = len(self.raw_input[0])*int(self.raw_input[0][0].shape[0])
            raw_input_numel_0 = len(self.raw_input[0])*int(self.raw_input[0][0].numel())
            raw_input_numel_1 = len(self.raw_input[1])*int(self.raw_input[1][0].numel())
            raw_out_numel = len(self.raw_out[0])*int(self.raw_out[0][0].numel())
        elif self.raw_input[0].type()=='torch.FloatTensor':
            self.calib_size = int(self.raw_input[0].shape[0])
            self.calib_batch_size = int(self.raw_input[0].shape[0])
            raw_input_numel_0 = self.raw_input[0].numel()
            raw_input_numel_1 = self.raw_input[1].numel()
            raw_out_numel = self.raw_out.numel()
            
        # self.calib_size = int(self.raw_input[0][0].shape[0])
        # self.calib_batch_size = int(self.raw_input[0].shape[0])
        # self.calib_batch_size = 512
        while True:
            numel = ((raw_input_numel_0 + raw_input_numel_1 + 2*raw_out_numel)/self.calib_size*self.calib_batch_size) # number of parameters on GPU
            self.parallel_eq_n = int((3*1024*1024*1024/4)//numel)
            if self.parallel_eq_n <= 1:
                self.calib_need_batching = True
                self.calib_batch_size //= 2
            else:
                break
    
    def _get_padding_parameters_with_shape(self):
        assert self.A_shape is not None,f"You should define self.A_shape before run quant_forward for {self}"
        A_shape = self.A_shape
        B_shape = self.B_shape
        
        self.n_G_A = A_shape[1]
        self.n_G_B = B_shape[1]
        super()._get_padding_parameters_with_shape()
        
    def _get_padding_parameters(self, A, B):
        """
        We adopt a head-wise quantization here
        """
        self.n_G_A = A.shape[1]
        self.n_G_B = B.shape[1]
        super()._get_padding_parameters(A,B)
    
    def _initialize_intervals(self):
        # pad A and B for future quantization
        if not self.revised:
            self._get_padding_parameters(self.raw_input[0], self.raw_input[1]) # put it here because hessian does not use calibration step 1

            # initialize intervals with minmax intervals
            tmp_A_intervals = []
            tmp_B_intervals = []
            for b_st in range(0,self.calib_size,self.calib_batch_size):
                b_ed = min(self.calib_size, b_st+self.calib_batch_size)
                A, B = self.raw_input[0][b_st:b_ed].to(self.device), self.raw_input[1][b_st:b_ed].to(self.device)
                if self.init_layerwise:
                    A_interval = (A.abs().max()/(self.A_qmax-0.5)).detach().view(1,1,1,1,1,1,1).repeat(1,self.n_G_A,1,self.n_V_A,1,self.n_H_A,1)
                    B_interval = (B.abs().max()/(self.B_qmax-0.5)).detach().view(1,1,1,1,1,1,1).repeat(1,self.n_G_B,1,self.n_V_B,1,self.n_H_B,1)
                else:
                    A_pad = F.pad(A, [0,self.pad_cols_A,0,self.pad_rows_A,0,self.pad_groups_A]).unsqueeze(0).view(1,-1,self.n_G_A,self.crb_groups_A,self.n_V_A,self.crb_rows_A,self.n_H_A,self.crb_cols_A)
                    B_pad = F.pad(B, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B]).unsqueeze(0).view(1,-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)
                    A_interval=(A_pad.abs().amax([0,1,3,5,7], keepdim=True)/(self.A_qmax-0.5)).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1
                    B_interval=(B_pad.abs().amax([0,1,3,5,7], keepdim=True)/(self.B_qmax-0.5)).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1
                tmp_A_intervals.append(A_interval)
                tmp_B_intervals.append(B_interval)
            self.A_interval = torch.cat(tmp_A_intervals, dim=0).amax(0, keepdim=True)
            self.B_interval = torch.cat(tmp_B_intervals, dim=0).amax(0, keepdim=True)
        else:
            self._get_padding_parameters(self.raw_input[0], self.raw_input[1]) # put it here because hessian does not use calibration step 1

            # initialize intervals with minmax intervals
            tmp_A_intervals = [] 
            tmp_B_intervals = []
            tmp_A_zeros = []
            tmp_B_zeros = []
            for b_st in range(0,self.calib_size,self.calib_batch_size):
                b_ed = min(self.calib_size, b_st+self.calib_batch_size)
                A, B = self.raw_input[0][b_st:b_ed].to(self.device), self.raw_input[1][b_st:b_ed].to(self.device)
                A_pad = F.pad(A, [0,self.pad_cols_A,0,self.pad_rows_A,0,self.pad_groups_A]).unsqueeze(0).view(1,-1,self.n_G_A,self.crb_groups_A,self.n_V_A,self.crb_rows_A,self.n_H_A,self.crb_cols_A)
                B_pad = F.pad(B, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B]).unsqueeze(0).view(1,-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)
                if self.sym:
                    A_interval=(A_pad.abs().amax([0,1,3,5,7], keepdim=True)/(self.A_qmax-0.5)).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1
                    B_interval=(B_pad.abs().amax([0,1,3,5,7], keepdim=True)/(self.B_qmax-0.5)).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1
                else:
                    A_max = A_pad.amax([0,1,3,5,7], keepdim=True)
                    A_min = A_pad.amin([0,1,3,5,7], keepdim=True)
                    B_max = B_pad.amax([0,1,3,5,7], keepdim=True)
                    B_min = B_pad.amin([0,1,3,5,7], keepdim=True)
                    A_interval=((A_max-A_min)/self.A_qmax).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1
                    B_interval=((B_max-B_min)/self.B_qmax).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1
                    A_zero = torch.round(-A_min.squeeze(0) / A_interval)
                    B_zero = torch.round(-B_min.squeeze(0) / B_interval)
                    tmp_A_zeros.append(A_zero)
                    tmp_B_zeros.append(B_zero)
                tmp_A_intervals.append(A_interval)
                tmp_B_intervals.append(B_interval)
            self.A_interval = torch.cat(tmp_A_intervals, dim=0).amax(0, keepdim=True)
            self.B_interval = torch.cat(tmp_B_intervals, dim=0).amax(0, keepdim=True)
            if self.sym:
                self.A_zero = torch.zeros_like(self.A_interval)
                self.B_zero = torch.zeros_like(self.B_interval)
                self.A_qmax = 2**(self.A_bit-1)
                self.B_qmax = 2**(self.B_bit-1)
                self.A_climp_max = self.A_qmax-1
                self.A_climp_min = -self.A_qmax
                self.B_climp_max = self.B_qmax-1
                self.B_climp_min = -self.B_qmax
            else:
                self.A_zero = torch.cat(tmp_A_zeros, dim=0).amax(0, keepdim=True)
                self.B_zero = torch.cat(tmp_B_zeros, dim=0).amax(0, keepdim=True)
                self.A_qmax = 2**(self.A_bit)-1
                self.B_qmax = 2**(self.B_bit)-1
                self.A_climp_max = self.A_qmax
                self.A_climp_min = 0
                self.B_climp_max = self.B_qmax
                self.B_climp_min = 0

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None, dim=-1, raw_grad=None):
        """
        tensor_raw: *, features, *
        tensor_sim: *, features, *
        similarity: *
        It's your job to calculate mean on non-feature * dims!

        Similarity without inherent feature structure is more welcome to parallelism.
        """
        if metric == "cosine":
            similarity = F.cosine_similarity(tensor_raw, tensor_sim, dim=dim) # should only support dim=-1 and cannot be paralleled
        elif metric == "pearson":
            similarity = F.cosine_similarity(tensor_raw-torch.mean(tensor_raw,dim=dim,keepdim=True), tensor_sim-torch.mean(tensor_sim,dim=dim,keepdim=True), dim=dim) # should only support dim=-1 and cannot be paralleled
            # a quick implementation of pearson similarity
            # tensor_raw: 1,B,H,dim1,dim3
            # tensor_sim: parallel_eq_n,B,H,dim1,dim3
            # parallel_eq_n,B,H,dim1,dim3 = tensor_sim.shape
            # tensor_sim = tensor_sim.view(parallel_eq_n,B,-1)
            # tensor_raw = tensor_raw.view(1,B,-1)
            # tensor_sim_mean = tensor_sim.mean(dim=[1,2],keepdim=True)
            # tensor_raw_mean = tensor_raw.mean(dim=[1,2],keepdim=True)
            # similarity = F.cosine_similarity(tensor_raw-tensor_raw_mean,tensor_sim-tensor_sim_mean,dim=-1) # shape: parallel_eq_n,B
            # similarity = similarity.reshape(parallel_eq_n,B,1,1) # restore two dims
        else:
            if metric == "L1_norm":
                similarity = -torch.abs(tensor_raw - tensor_sim)
            elif metric == "L2_norm":
                similarity = -(tensor_raw - tensor_sim) ** 2
            elif metric == "linear_weighted_L2_norm":
                similarity = -tensor_raw.abs() * (tensor_raw - tensor_sim) ** 2
            elif metric == "square_weighted_L2_norm":
                similarity = -(tensor_raw * (tensor_raw - tensor_sim)) ** 2
            elif metric == "hessian":
                assert raw_grad != None, f"No raw_grad in PTQSLBatchingQuantMatMul!"
                raw_grad = raw_grad.reshape_as(tensor_raw)
                similarity = -(raw_grad * (tensor_raw - tensor_sim)) ** 2
            else:
                raise NotImplementedError(f"metric {metric} not implemented!")
            similarity = torch.mean(similarity, dim=dim)
        return similarity

    def _search_best_A_interval(self, A_interval_candidates):
        """
        Modularization of searching best interval
        """
        tmp_A_interval = self.A_interval.unsqueeze(0) # shape: 1,1,n_G,1,n_V,1,n_H,1
        # out-of-loop optimization
        for v, h in product(range(self.n_V_A), range(self.n_H_A)):
            A_zeros = []
            batch_similarities = [] # similarities, need to concatenate and calculate sum
            # cal = tqdm.tqdm(range(0, self.calib_size, self.calib_batch_size), total=self.calib_size//self.calib_batch_size, desc='search_A_matmul')
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                A = self.raw_input[0][b_st:b_ed].to(self.device)
                A_pad = F.pad(A, [0,self.pad_cols_A,0,self.pad_rows_A,0,self.pad_groups_A]).unsqueeze(0).view(1,-1,self.n_G_A,self.crb_groups_A,self.n_V_A,self.crb_rows_A,self.n_H_A,self.crb_cols_A)
                B = self.raw_input[1][b_st:b_ed].to(self.device)
                B_sim = self.quant_input_B(B).unsqueeze(0) # shape: 1,b,H,dim2,dim3
                raw_out = self.raw_out[b_st:b_ed].unsqueeze(0).to(self.device)
                raw_grad = self.raw_grad[b_st:b_ed].to(self.device)
                similarities = []
                for p_st in range(0, self.eq_n, self.parallel_eq_n):
                    p_ed = min(self.eq_n,p_st+self.parallel_eq_n)
                    # quantize A
                    cur_A_interval = tmp_A_interval.repeat(p_ed-p_st,1,1,1,1,1,1,1)
                    cur_A_interval[:,:,:,:,v:v+1,:,h:h+1,:] = A_interval_candidates[p_st:p_ed,:,:,:,v:v+1,:,h:h+1,:]
                    if self.revised:
                        A_min = A_pad.amin([0,1,3,5,7], keepdim=True)
                        A_zero = torch.round(-A_min / cur_A_interval)
                        A_zeros.append(A_zero)
                        A_sim = torch.clamp(torch.round(A_pad / cur_A_interval) + A_zero, self.A_climp_min, self.A_climp_max)
                        A_sim = cur_A_interval * (A_sim - A_zero)
                    else:
                        A_sim = (A_pad/cur_A_interval).round_().clamp_(-self.A_qmax,self.A_qmax-1).mul_(cur_A_interval)
                    A_sim = A_sim.view(p_ed-p_st,-1,A.shape[1]+self.pad_groups_A,A.shape[2]+self.pad_rows_A,A.shape[3]+self.pad_cols_A) # shape: parallel_eq_n,B,H*,dim1*,dim2* (* stand for padding)
                    A_sim = A_sim[:,:,:A.shape[1],:A.shape[2],:A.shape[3]] # shape: parallel_eq_n,b,H,dim1,dim2
                    # quantize B, this quantization is optimized out of loop    
                    # calculate similarity and store them
                    out_sim = A_sim @ B_sim # shape: parallel_eq_n,B,H,dim1,dim3
                    similarity = self._get_similarity(raw_out, out_sim, self.metric, raw_grad=raw_grad) # shape: parallel_eq_n,b,H,dim1
                    similarity = similarity.mean([3]) # shape: parallel_eq_n,b,H (remaining mean operation will be done later on)
                    similarity = similarity.sum(dim=1, keepdim=True) # shape: parallel_eq_n,1,H
                    similarities.append(similarity)
                # calculate best similarity for this block
                similarities = torch.cat(similarities, 0) # shape: eq_n,1,H
                batch_similarities.append(similarities)
            batch_similarities = torch.cat(batch_similarities, dim=1).sum(dim=1, keepdim=False) #shape: eq_n,H
            batch_similarities = F.pad(batch_similarities, [0,self.pad_groups_A]).view(self.eq_n,self.n_G_A,self.crb_groups_A).mean(-1) # shape: eq_n, n_G_A
            best_index = torch.argmax(batch_similarities, dim=0, keepdim=False).view(1,1,-1,1,1,1,1,1)
            tmp_A_interval[:,:,:,:,v:v+1,:,h:h+1,:] = torch.gather(A_interval_candidates[:,:,:,:,v:v+1,:,h:h+1,:],dim=0,index=best_index)
            if self.revised:
                A_zeros = torch.cat(A_zeros, dim=0)
                A_zeros = torch.gather(A_zeros, dim=0, index=best_index) 
        self.A_interval = tmp_A_interval.squeeze(0)
        if self.revised:
            self.A_zero = A_zeros.squeeze(0)


    def _search_best_B_interval(self, B_interval_candidates):
        """
        Modularization of searching best interval
        """
        tmp_B_interval = self.B_interval.unsqueeze(0) # shape: 1,1,n_G,1,n_V,1,n_H,1
        # out-of-loop optimization
        for v, h in product(range(self.n_V_B), range(self.n_H_B)):
            B_zeros = []
            batch_similarities = [] # similarities, need to concatenate and calculate sum
            # cal = tqdm.tqdm(range(0, self.calib_size, self.calib_batch_size), total=self.calib_size//self.cali    b_batch_size, desc='search_B_matmul')
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                A = self.raw_input[0][b_st:b_ed].to(self.device)
                A_sim = self.quant_input_A(A).unsqueeze(0) # shape: 1,B,H,dim1,dim2
                B = self.raw_input[1][b_st:b_ed].to(self.device)
                B_pad = F.pad(B, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B]).unsqueeze(0).view(1,-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)
                raw_out = self.raw_out[b_st:b_ed].unsqueeze(0).to(self.device)
                raw_grad = self.raw_grad[b_st:b_ed].to(self.device)
                similarities = []
                for p_st in range(0, self.eq_n, self.parallel_eq_n):
                    p_ed = min(self.eq_n,p_st+self.parallel_eq_n)
                    # quantize A, this quantization is optimized out of loop
                    # quantize B
                    cur_B_interval = tmp_B_interval.repeat(p_ed-p_st,1,1,1,1,1,1,1)
                    cur_B_interval[:,:,:,:,v:v+1,:,h:h+1,:] = B_interval_candidates[p_st:p_ed,:,:,:,v:v+1,:,h:h+1,:]
                    if self.revised:
                        B_min = B_pad.amin([0,1,3,5,7], keepdim=True)
                        B_zero = torch.round(-B_min / cur_B_interval)
                        B_zeros.append(B_zero)
                        B_sim = torch.clamp(torch.round(B_pad / cur_B_interval) + B_zero, self.B_climp_min, self.B_climp_max)
                        B_sim = cur_B_interval * (B_sim - B_zero)
                    else:
                        B_sim = (B_pad/cur_B_interval).round_().clamp_(-self.B_qmax,self.B_qmax-1).mul_(cur_B_interval)
                    B_sim = B_sim.view(p_ed-p_st,-1,B.shape[1]+self.pad_groups_B,B.shape[2]+self.pad_rows_B,B.shape[3]+self.pad_cols_B) # shape: parallel_eq_n,b,H*,dim2*,dim3* (* stand for padding)
                    B_sim = B_sim[:,:,:B.shape[1],:B.shape[2],:B.shape[3]] # shape: parallel_eq_n,b,H,dim2,dim3
                    # calculate similarity and store them
                    out_sim = A_sim @ B_sim # shape: parallel_eq_n,b,H,dim1,dim3
                    similarity = self._get_similarity(raw_out, out_sim, self.metric, raw_grad=raw_grad) # shape: parallel_eq_n,b,H,dim1
                    similarity = similarity.mean([3]) # shape: parallel_eq_n,b,H (remaining mean operation will be done later on)
                    similarity = similarity.sum(dim=1, keepdim=True) # shape: parallel_eq_n,1,H
                    similarities.append(similarity) 
                # calculate best similarity for this block
                similarities = torch.cat(similarities, 0) # shape: eq_n,1,H
                batch_similarities.append(similarities)
            batch_similarities = torch.cat(batch_similarities, dim=1).sum(dim=1, keepdim=False) #shape: eq_n,H
            batch_similarities = F.pad(batch_similarities, [0,self.pad_groups_B]).view(self.eq_n,self.n_G_B,self.crb_groups_B).mean(-1) # shape: eq_n, n_G_B
            best_index = torch.argmax(batch_similarities, dim=0, keepdim=False).view(1,1,-1,1,1,1,1,1)
            tmp_B_interval[:,:,:,:,v:v+1,:,h:h+1,:] = torch.gather(B_interval_candidates[:,:,:,:,v:v+1,:,h:h+1,:],dim=0,index=best_index)
            if self.revised:
                B_zeros = torch.cat(B_zeros, dim=0)
                B_zeros = torch.gather(B_zeros, dim=0, index=best_index) 
        self.B_interval = tmp_B_interval.squeeze(0)
        if self.revised:
            self.B_zero = B_zeros.squeeze(0)

    def _random_sample_calib_raw_data(self):
        self.calib_size = 32
        if isinstance(self.raw_input[0], list):
            inter_batch_num = self.calib_size // self.raw_input[0][0].shape[0] # 32 // batch_size(4) = 8
            random_ind = torch.randperm(len(self.raw_input[0]))[:inter_batch_num]
            self.raw_input[0] = torch.cat([self.raw_input[0][i] for i in random_ind],dim=0)
            self.raw_input[1] = torch.cat([self.raw_input[1][i] for i in random_ind],dim=0)
            self.raw_out = torch.cat([self.raw_out[i] for i in random_ind],dim=0)
            self.raw_grad = torch.cat([self.raw_grad[i] for i in random_ind],dim=0)
        else:
            random_ind = torch.randperm(len(self.raw_input))[:self.calib_size]
            self.raw_input[0] = torch.stack([self.raw_input[0][i] for i in random_ind])
            self.raw_input[1] = torch.stack([self.raw_input[1][i] for i in random_ind])
            self.raw_out = torch.stack([self.raw_out[i] for i in random_ind])
            self.raw_grad = torch.stack([self.raw_grad[i] for i in random_ind])
    
    # def _random_sample_calib_raw_data(self):
        # random_ind = torch.randperm(self.calib_size)[:self.calib_batch_size]
        # self.raw_input[0] = self.raw_input[0][random_ind]
        # self.raw_input[1] = self.raw_input[1][random_ind]
        # self.calib_size = self.calib_batch_size 
            
    def calibration_step2(self, device, group_num=None):
        self.device = device
        self._initialize_calib_parameters()
        self._random_sample_calib_raw_data()
        self._initialize_intervals()
        A_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.device).view(-1,1,1,1,1,1,1,1) * self.A_interval.unsqueeze(0)
        B_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.device).view(-1,1,1,1,1,1,1,1) * self.B_interval.unsqueeze(0)
        for e in range(self.search_round):
            # search for best A interval
            self._search_best_A_interval(A_interval_candidates)
            # search for best B interval
            self._search_best_B_interval(B_interval_candidates)
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad
    
    def quant_input_A(self, x):
        A_interval = self.A_interval.reshape(1, self.A_interval.shape[1], 1, 1)
        if not self.revised:
            x = (x/A_interval).round_().clamp(-self.A_qmax,self.A_qmax-1).mul_(A_interval)
        else:
            A_zero = self.A_zero.reshape(1, self.A_zero.shape[1], 1, 1)
            x = torch.clamp(torch.round(x / A_interval) + A_zero, self.A_climp_min, self.A_climp_max)
            x = A_interval * (x - A_zero)
        return x
    
    def quant_input_B(self, x):
        B_interval = self.B_interval.reshape(1, self.B_interval.shape[1], 1, 1)
        if not self.revised:
            x = (x/B_interval).round_().clamp(-self.B_qmax,self.B_qmax-1).mul_(B_interval)
        else:
            B_zero = self.B_zero.reshape(1, self.B_zero.shape[1], 1, 1)
            x = torch.clamp(torch.round(x / B_interval) + B_zero, self.B_climp_min, self.B_climp_max)
            x = B_interval * (x - B_zero)
        return x

class SoSPTQSLBatchingQuantMatMul(PTQSLBatchingQuantMatMul):
    def __init__(self, A_bit=8, B_bit=8, mode="raw", sym=False,
                 metric="L2_norm", search_round=1, eq_alpha=0.1, eq_beta=2, eq_n=100, parallel_eq_n=10,
                 n_G_A=1, n_V_A=1, n_H_A=1, n_G_B=1, n_V_B=1, n_H_B=1, init_layerwise=False,
                 split=None):
        super().__init__(A_bit=A_bit, B_bit=B_bit, mode=mode, sym=sym,
                         metric=metric, search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=parallel_eq_n, 
                         n_G_A=n_G_A, n_V_A=n_V_A, n_H_A=n_H_A, n_G_B=n_G_B, n_V_B=n_V_B, n_H_B=n_H_B, init_layerwise=init_layerwise)
        self.n_G_A = 1
        self.n_V_A = 1
        self.n_H_A = 1
        # with proper hardware implementation, we don't need to use a sign bit anymore
        self.A_qmax = 2**(self.A_bit-1)
        self.split = split
        if split != None:
            self.A_interval = self.split/(self.A_qmax-1)
        self.parallel_eq_n = parallel_eq_n
        self.sym = sym

    def quant_input_A(self, x):
        self.A_qmax = 2**(self.A_bit-1)
        x_high = (x.clamp(self.split, 1)*(self.A_qmax-1)).round_().clamp_(0,self.A_qmax-1)/(self.A_qmax-1)
        x_low = (x.clamp(0, self.split)/self.A_interval).round_().clamp_(0,self.A_qmax-1)*self.A_interval
        return x_high + x_low

    def _search_best_A_interval(self, split_candidates):
        batch_similarities = []
        # cal = tqdm.tqdm(range(0, self.calib_size, self.calib_batch_size), total=self.calib_size//self.calib_batch_size, desc='search_A_matmul')
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            A = self.raw_input[0][b_st:b_ed].unsqueeze(0).to(self.device)
            B = self.raw_input[1][b_st:b_ed].unsqueeze(0).to(self.device)
            B_sim = B
            raw_out = self.raw_out[b_st:b_ed].unsqueeze(0).to(self.device)
            raw_grad = self.raw_grad[b_st:b_ed].to(self.device)
            similarities = []
            for i in range(len(split_candidates)):
                # quantize A
                cur_A_interval = split_candidates[i]/(self.A_qmax-1)
                A_high = (A.clamp(split_candidates[i], 1)*(self.A_qmax-1)).round_().clamp_(0,self.A_qmax-1)/(self.A_qmax-1)
                A_low =( A.clamp(0, split_candidates[i])/cur_A_interval).round_().clamp_(0,self.A_qmax-1)*cur_A_interval
                A_sim = A_high + A_low # shape: 1,b,H,S,S
                # quantize B, this quantization is optimized out of loop
                # calculate similarity and store them (dim1=dim2=S, dim3=W)
                out_sim = A_sim @ B_sim # shape: 1,b,H,dim1,dim3
                similarity = self._get_similarity(raw_out, out_sim, self.metric, raw_grad=raw_grad) # shape: parallel_eq_n,b,H,dim1
                similarity = similarity.mean([2,3]) # shape: parallel_eq_n, b
                similarity = similarity.sum(dim=1,keepdim=True) # parallel_eq_n, 1
                similarities.append(similarity)
            # calculate best similarity for this block
            similarities = torch.cat(similarities, 0) # shape: eq_n, 1
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=1).sum(dim=1, keepdim=False) #shape: eq_n
        best_index = torch.argmax(batch_similarities, dim=0, keepdim=False)
        self.split = split_candidates[best_index]
        self.A_interval = self.split/(self.A_qmax-1)
        # debugging
        # print(f"best split: {self.split}")

    def calibration_step2(self,device, group_num=None):
        self.device = device
        self._initialize_calib_parameters()
        self._initialize_intervals()
        A_split_candidates = torch.tensor([2**(-i) for i in range(20)]).to(self.device)
        B_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.device).view(-1,1,1,1,1,1,1,1) * self.B_interval.unsqueeze(0)
        for e in range(self.search_round):
            # search for best A interval
            self._search_best_A_interval(A_split_candidates)
            # search for best B interval
            self._search_best_B_interval(B_interval_candidates)
        self.calibrated = True
        print(f"best split: {self.split}")
        del self.raw_input, self.raw_out, self.raw_grad
    
    def quant_forward(self, A, B):
        # assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        assert self.A_interval is not None,f"You should define self.A_interval before run quant_forward for {self}"
        A_sim=self.quant_input_A(A)
        B_sim=self.quant_input_B(B)
        self.A_quant = A_sim
        self.B_quant = B_sim
        out=A_sim@B_sim
        return out
    

class LISPTQSLBatchingQuantMatMul(PTQSLBatchingQuantMatMul):
    def __init__(self, A_bit=8, B_bit=8, mode="raw",
                 metric="L2_norm", search_round=1, eq_alpha=0.1, eq_beta=2, eq_n=100, parallel_eq_n=10,
                 n_G_A=1, n_V_A=1, n_H_A=1, n_G_B=1, n_V_B=1, n_H_B=1, init_layerwise=False,
                 split=None):
        super().__init__(A_bit=A_bit, B_bit=B_bit, mode=mode, 
                         metric=metric, search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=parallel_eq_n, 
                         n_G_A=n_G_A, n_V_A=n_V_A, n_H_A=n_H_A, n_G_B=n_G_B, n_V_B=n_V_B, n_H_B=n_H_B, init_layerwise=init_layerwise)
        self.n_G_A = 1
        self.n_V_A = 1
        self.n_H_A = 1
        # with proper hardware implementation, we don't need to use a sign bit anymore
        self.A_qmax = 2**(self.A_bit-1)

    def calibration_step2(self,device,timestep = None):
        self.device = device
        self._initialize_calib_parameters()
        self._initialize_intervals()
        A_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.device).view(-1,1,1,1,1,1,1,1) * self.A_interval.unsqueeze(0)
        B_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.device).view(-1,1,1,1,1,1,1,1) * self.B_interval.unsqueeze(0)
        for e in range(self.search_round):
            # search for best A interval
            self._search_best_A_interval(A_interval_candidates)
            # search for best B interval
            self._search_best_B_interval(B_interval_candidates)
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad
    
    def log_round(self,x):
        x_log_floor = x.log2().floor()
        big = x_log_floor
        extra_mask = (x - 2**big) >= 2**(big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big
    
    def quant_forward(self, A, B):
        # assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        assert self.A_interval is not None,f"You should define self.A_interval before run quant_forward for {self}"
        A_sim=self.quant_input_A(A)
        B_sim=self.quant_input_B(B)
        
        # LIS
        exp_int, exp_int_sum = self.int_softmax(A_sim, self.A_interval)
        softmax_out = torch.round(exp_int_sum / exp_int)
        rounds = self.log_round(softmax_out)
        mask = rounds >= 2**self.A_bit
        qlog = torch.clamp(rounds, 0, 2**self.A_bit - 1)
        deq_softmax = 2**(-qlog)
        deq_softmax[mask] = 0
        A_sim = deq_softmax
        
        self.A_quant = A_sim
        self.B_quant = B_sim
        out=A_sim@B_sim
        return out
    
    def int_softmax(self, x, scaling_factor):

        def int_polynomial(x_int, scaling_factor):
            coef = [0.35815147, 0.96963238, 1.]  # ax**2 + bx + c
            coef[1] /= coef[0]
            coef[2] /= coef[0]
            b_int = torch.floor(coef[1] / scaling_factor)
            c_int = torch.floor(coef[2] / scaling_factor**2)
            z = x_int + b_int
            z = x_int * z
            z = z + c_int
            scaling_factor = coef[0] * scaling_factor**2
            return z, scaling_factor

        def int_exp(x_int, scaling_factor):
            x0 = -0.6931  # -ln2
            n = 30  # sufficiently large integer
            x0_int = torch.floor(x0 / scaling_factor)
            x_int = torch.max(x_int, n * x0_int)
            q = torch.floor(x_int / x0_int)
            r = x_int - x0_int * q
            exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
            exp_int = torch.clamp(torch.floor(exp_int * 2**(n - q)), min=0)
            scaling_factor = exp_scaling_factor / 2**n
            return exp_int, scaling_factor

        scaling_factor = scaling_factor.reshape(1, scaling_factor.shape[1], 1, 1)
        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        # x_int_max = x_int.amax(dim=(-1, -2), keepdim=True)
        # x_int_max = x_int.max()
        x_int = x_int - x_int_max
        exp_int, exp_scaling_factor = int_exp(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        return exp_int, exp_int_sum
    
class SoLPTQSLBatchingQuantMatMul(PTQSLBatchingQuantMatMul):
    def __init__(self, A_bit=8, B_bit=8, mode="raw",
                 metric="L2_norm", search_round=1, eq_alpha=0.1, eq_beta=2, eq_n=100, parallel_eq_n=10,
                 n_G_A=1, n_V_A=1, n_H_A=1, n_G_B=1, n_V_B=1, n_H_B=1, init_layerwise=False,
                 split=None):
        super().__init__(A_bit=A_bit, B_bit=B_bit, mode=mode, 
                         metric=metric, search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=parallel_eq_n, 
                         n_G_A=n_G_A, n_V_A=n_V_A, n_H_A=n_H_A, n_G_B=n_G_B, n_V_B=n_V_B, n_H_B=n_H_B, init_layerwise=init_layerwise)
        self.n_G_A = 1
        self.n_V_A = 1
        self.n_H_A = 1
        # with proper hardware implementation, we don't need to use a sign bit anymore
        self.A_qmax = 2**(self.A_bit-1)
        self.split = split
        if split != None:
            self.A_interval = self.split/(self.A_qmax-1)
            
    def quant_input_A(self, x):
        rounds = torch.round(-1 * x.log2())
        softmax_mask = rounds >= 2**self.A_bit
        x = torch.clamp(rounds, 0, 2**self.A_bit - 1)
        x = 2**(-1 * x)
        x[softmax_mask] = 0
        return x
        
    def calibration_step2(self,device,timestep = None):
        self.device = device
        self._initialize_calib_parameters()
        self._initialize_intervals()
        B_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.device).view(-1,1,1,1,1,1,1,1) * self.B_interval.unsqueeze(0)
        self._search_best_B_interval(B_interval_candidates)
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad
    
    def quant_forward(self, A, B):
        # assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        assert self.A_interval is not None,f"You should define self.A_interval before run quant_forward for {self}"
        A_sim = self.quant_input_A(A)
        B_sim=self.quant_input_B(B)
        # weight_plot(A_sim[:,0,:,:],"last_sol_quantized_A")
        self.A_quant = A_sim
        self.B_quant = B_sim
        out=A_sim@B_sim
        return out
    
# class SoLog(nn.Module):
#     def __init__(self, A_bit=8, B_bit=8, mode="raw"):
#         super().__init__()
#         self.A_bit=A_bit
#         self.B_bit=B_bit
#         self.A_interval=None
#         self.B_interval=None
#         self.A_qmax=2**(self.A_bit-1)
#         self.B_qmax=2**(self.B_bit-1)
#         self.mode=mode
#         self.raw_input = None
#         self.raw_out = None
        
#     def forward(self, out):
#         if self.mode=='raw':
#             out= out.softmax(dim=-1)
#         else:
#             out=self.quant_forward(out)
#         return out
    
#     def quant_forward(self, x):
#         rounds = torch.round(-1 * x.log2())
#         softmax_mask = rounds >= 2**self.A_bit
#         x = torch.clamp(rounds, 0, 2**self.A_bit - 1)
#         x = 2**(-1 * x)
#         x[softmax_mask] = 0
#         return x

class TimewiseSoSPTQSLBatchingQuantMatMul(SoSPTQSLBatchingQuantMatMul):
    def __init__(self, A_bit=8, B_bit=8, mode="raw", sym=False,
                 metric="L2_norm", search_round=1, eq_alpha=0.1, eq_beta=2, eq_n=100, parallel_eq_n=10,
                 n_G_A=1, n_V_A=1, n_H_A=1, n_G_B=1, n_V_B=1, n_H_B=1, init_layerwise=False,
                 split=None, timegroup_num=25, total_timestep = 250, revised = False):
        super().__init__(A_bit=A_bit, B_bit=B_bit, mode=mode, sym = sym,
                         metric=metric, search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=parallel_eq_n, 
                         n_G_A=n_G_A, n_V_A=n_V_A, n_H_A=n_H_A, n_G_B=n_G_B, n_V_B=n_V_B, n_H_B=n_H_B, init_layerwise=init_layerwise)
        self.A_interval = dict()
        self.timegroup_num = timegroup_num
        self.total_timestep = total_timestep
        self.split = split
        self.sym = sym
        self.revised = revised
        if split != None:
            self.A_interval = self.split/(self.A_qmax-1)
        # self.raw_input[0] : dict
    
    def _initialize_calib_parameters(self):
        """ 
        set parameters for feeding calibration data
        """
        # self.raw_input[0] is dictionary
        if isinstance(self.raw_input[1], dict):
            self.calib_size = int(self.raw_input[1][0].shape[0])
            self.calib_batch_size = int(self.raw_input[1][0].shape[0])
            raw_input_numel_1 = self.raw_input[1][0].numel()
            raw_out_numel = self.raw_out[0].numel()
        elif isinstance(self.raw_input[1], list):
            self.calib_size = len(self.raw_input[1])*int(self.raw_input[1][0].shape[0])
            self.calib_batch_size = len(self.raw_input[1])*int(self.raw_input[1][0].shape[0])
            raw_input_numel_1 = len(self.raw_input[1])*int(self.raw_input[1][0].numel())
            raw_out_numel = len(self.raw_out[0])*int(self.raw_out[0][0].numel())
        elif self.raw_input[1].type()=='torch.FloatTensor':
            self.calib_size = int(self.raw_input[1].shape[0])
            self.calib_batch_size = int(self.raw_input[1].shape[0])
            raw_input_numel_1 = self.raw_input[1].numel()
            raw_out_numel = self.raw_out.numel()

        # self.calib_size = int(self.raw_input[1].shape[0])
        # self.calib_batch_size = int(self.raw_input[0].shape[0])
        # self.calib_batch_size = 512
        while True:
            numel = ((self.raw_input[0][0].numel() + raw_input_numel_1 + 2*raw_out_numel)/self.calib_size*self.calib_batch_size) # number of parameters on GPU
            self.parallel_eq_n = int((3*1024*1024*1024/4)//numel)
            if self.parallel_eq_n <= 1:
                self.calib_need_batching = True
                self.calib_batch_size //= 2
            else:
                break
        
    def _initialize_intervals(self):
        # pad A and B for future quantization
        if isinstance(self.raw_input[1], dict):
            raw_input_B = self.raw_input_1
        self._get_padding_parameters(self.raw_input[0][0], raw_input_B[1]) # put it here because hessian does not use calibration step 1

        # initialize intervals with minmax intervals
        tmp_A_intervals = dict()
        tmp_B_intervals = []
        tmp_B_zeros = []
    
        if not self.revised:
            for b_st in range(0,self.calib_size,self.calib_batch_size):
                b_ed = min(self.calib_size, b_st+self.calib_batch_size)
                B = raw_input_B[1][b_st:b_ed].to(self.device)
                if self.init_layerwise:
                    B_interval = (B.abs().max()/(self.B_qmax-0.5)).detach().view(1,1,1,1,1,1,1).repeat(1,self.n_G_B,1,self.n_V_B,1,self.n_H_B,1)
                else:
                    B_pad = F.pad(B, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B]).unsqueeze(0).view(1,-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)
                    B_interval=(B_pad.abs().amax([0,1,3,5,7], keepdim=True)/(self.B_qmax-0.5)).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1
                tmp_B_intervals.append(B_interval)
            self.B_interval = torch.cat(tmp_B_intervals, dim=0).amax(0, keepdim=True)
        else:
            for b_st in range(0,self.calib_size,self.calib_batch_size):
                b_ed = min(self.calib_size, b_st+self.calib_batch_size)
                B = raw_input_B[1][b_st:b_ed].to(self.device)
                B_pad = F.pad(B, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B]).unsqueeze(0).view(1,-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)
                if self.sym:
                    B_interval=(B_pad.abs().amax([0,1,3,5,7], keepdim=True)/(self.B_qmax-0.5)).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1
                else:
                    B_max = B_pad.amax([0,1,3,5,7], keepdim=True)
                    B_min = B_pad.amin([0,1,3,5,7], keepdim=True)
                    B_interval=((B_max-B_min)/self.B_qmax).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1
                    B_zero = torch.round(-B_min.squeeze(0) / B_interval)
                    tmp_B_zeros.append(B_zero)
                tmp_B_intervals.append(B_interval)
            self.B_interval = torch.cat(tmp_B_intervals, dim=0).amax(0, keepdim=True)
            if self.sym:
                self.B_zero = torch.zeros_like(self.B_interval)
                self.B_qmax = 2**(self.B_bit-1)
                self.B_climp_max = self.B_qmax-1
                self.B_climp_min = -self.B_qmax
            else:
                self.B_zero = torch.cat(tmp_B_zeros, dim=0).amax(0, keepdim=True)
                self.B_qmax = 2**(self.B_bit)-1
                self.B_climp_max = self.B_qmax
                self.B_climp_min = 0
        
        self.A_interval = dict()
        for i in range(self.timegroup_num):
            tmp_A_intervals[i] = []
            for b_st in range(0,self.calib_size,self.calib_batch_size):
                b_ed = min(self.calib_size, b_st+self.calib_batch_size)
                A = self.raw_input[0][i][b_st:b_ed].to(self.device)
                if self.init_layerwise:
                    A_interval = (A.abs().max()/(self.A_qmax-0.5)).detach().view(1,1,1,1,1,1,1).repeat(1,self.n_G_A,1,self.n_V_A,1,self.n_H_A,1)
                else:
                    A_pad = F.pad(A, [0,self.pad_cols_A,0,self.pad_rows_A,0,self.pad_groups_A]).unsqueeze(0).view(1,-1,self.n_G_A,self.crb_groups_A,self.n_V_A,self.crb_rows_A,self.n_H_A,self.crb_cols_A)
                    A_interval=(A_pad.abs().amax([0,1,3,5,7], keepdim=True)/(self.A_qmax-0.5)).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1
                tmp_A_intervals[i].append(A_interval)
            self.A_interval[i] = torch.cat(tmp_A_intervals[i], dim=0).amax(0, keepdim=True)
    
    def quant_input_A(self, x, timegroup_ind):
        # timegroup_ind: integer
        self.A_qmax = 2**(self.A_bit-1)
        x_high = (x.clamp(self.split[timegroup_ind], 1)*(self.A_qmax-1)).round_().clamp_(0,self.A_qmax-1)/(self.A_qmax-1)
        x_low = (x.clamp(0, self.split[timegroup_ind])/self.A_interval[timegroup_ind]).round_().clamp_(0,self.A_qmax-1)*self.A_interval[timegroup_ind]
        return x_high + x_low
    
    def quant_input_A_forward(self, x, timegroup_ind): # 수정 중
        # timegroup_ind: list (shape : len(x) ( = len(timegroup_ind)))
        x_high = []
        x_low = []
        self.A_qmax = 2**(self.A_bit-1)
        for i in range(len(timegroup_ind)):
            x_high.append((x[i].clamp(self.split[timegroup_ind[i]], 1)*(self.A_qmax-1)).round_().clamp_(0,self.A_qmax-1)/(self.A_qmax-1))
            x_low.append((x[i].clamp(0, self.split[timegroup_ind[i]])/self.A_interval[timegroup_ind[i]]).round_().clamp_(0,self.A_qmax-1)*self.A_interval[timegroup_ind[i]])
        x_high = torch.stack(x_high)
        x_low = torch.stack(x_low)  
        return x_high + x_low

    def _search_best_A_interval(self, split_candidates, timegroup_ind):
        batch_similarities = []
        # cal = tqdm.tqdm(range(0, self.calib_size, self.calib_batch_size), total=self.calib_size//self.calib_batch_size, desc='search_A_matmul')
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            A = self.raw_input[0][timegroup_ind][b_st:b_ed].unsqueeze(0).to(self.device)
            B = self.raw_input[1][timegroup_ind][b_st:b_ed].unsqueeze(0).to(self.device)
            B_sim = B
            raw_out = self.raw_out[timegroup_ind][b_st:b_ed].unsqueeze(0).to(self.device)
            raw_grad = self.raw_grad[timegroup_ind][b_st:b_ed].to(self.device)
            similarities = []
            for i in range(len(split_candidates)):
                # quantize A
                cur_A_interval = split_candidates[i]/(self.A_qmax-1)
                A_high = (A.clamp(split_candidates[i], 1)*(self.A_qmax-1)).round_().clamp_(0,self.A_qmax-1)/(self.A_qmax-1)
                A_low =( A.clamp(0, split_candidates[i])/cur_A_interval).round_().clamp_(0,self.A_qmax-1)*cur_A_interval
                A_sim = A_high + A_low # shape: 1,b,H,S,S
                # quantize B, this quantization is optimized out of loop
                # calculate similarity and store them (dim1=dim2=S, dim3=W)
                out_sim = A_sim @ B_sim # shape: 1,b,H,dim1,dim3
                similarity = self._get_similarity(raw_out, out_sim, self.metric, raw_grad=raw_grad) # shape: parallel_eq_n,b,H,dim1
                similarity = similarity.mean([2,3]) # shape: parallel_eq_n, b
                similarity = similarity.sum(dim=1,keepdim=True) # parallel_eq_n, 1
                similarities.append(similarity)
            # calculate best similarity for this block
            similarities = torch.cat(similarities, 0) # shape: eq_n, 1
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=1).sum(dim=1, keepdim=False) #shape: eq_n
        best_index = torch.argmax(batch_similarities, dim=0, keepdim=False)
        self.split[timegroup_ind] = split_candidates[best_index]
        self.A_interval[timegroup_ind] = self.split[timegroup_ind]/(self.A_qmax-1)
        # debugging
        # print(f"best split: {self.split}")
        
    def _search_best_B_interval(self, B_interval_candidates):
        """
        Modularization of searching best interval
        """
        tmp_B_interval = self.B_interval.unsqueeze(0) # shape: 1,1,n_G,1,n_V,1,n_H,1
        # out-of-loop optimization
        for v, h in product(range(self.n_V_B), range(self.n_H_B)):
            B_zeros = []
            batch_similarities = [] # similarities, need to concatenate and calculate sum
            # cal = tqdm.tqdm(range(0, self.calib_size, self.calib_batch_size), total=self.calib_size//self.calib_batch_size, desc='search_B_matmul')
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                # timegroup_ind = 0
                # raw_input, raw_out, raw_grad = random_sample_from_dict()
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                A = self.raw_input_1[0][b_st:b_ed].to(self.device)
                A_sim = self.quant_input_A_forward(A, self.time_groups).unsqueeze(0) # shape: 1,B,H,dim1,dim2
                B = self.raw_input_1[1][b_st:b_ed].to(self.device)
                B_pad = F.pad(B, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B]).unsqueeze(0).view(1,-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)
                raw_out = self.raw_out_1[b_st:b_ed].unsqueeze(0).to(self.device)
                raw_grad = self.raw_grad_1[b_st:b_ed].to(self.device)
                similarities = []
                for p_st in range(0, self.eq_n, self.parallel_eq_n):
                    p_ed = min(self.eq_n,p_st+self.parallel_eq_n)
                    # quantize A, this quantization is optimized out of loop
                    # quantize B
                    cur_B_interval = tmp_B_interval.repeat(p_ed-p_st,1,1,1,1,1,1,1)
                    cur_B_interval[:,:,:,:,v:v+1,:,h:h+1,:] = B_interval_candidates[p_st:p_ed,:,:,:,v:v+1,:,h:h+1,:]
                    
                    if self.revised:
                        B_min = B_pad.amin([0,1,3,5,7], keepdim=True)
                        B_zero = torch.round(-B_min / cur_B_interval)
                        B_zeros.append(B_zero)
                        B_sim = torch.clamp(torch.round(B_pad / cur_B_interval) + B_zero, self.B_climp_min, self.B_climp_max)
                        B_sim = cur_B_interval * (B_sim - B_zero)
                    else:
                        B_sim = (B_pad/cur_B_interval).round_().clamp_(-self.B_qmax,self.B_qmax-1).mul_(cur_B_interval)
                    
                    B_sim = B_sim.view(p_ed-p_st,-1,B.shape[1]+self.pad_groups_B,B.shape[2]+self.pad_rows_B,B.shape[3]+self.pad_cols_B) # shape: parallel_eq_n,b,H*,dim2*,dim3* (* stand for padding)
                    B_sim = B_sim[:,:,:B.shape[1],:B.shape[2],:B.shape[3]] # shape: parallel_eq_n,b,H,dim2,dim3
                    # calculate similarity and store them
                    out_sim = A_sim @ B_sim # shape: parallel_eq_n,b,H,dim1,dim3
                    similarity = self._get_similarity(raw_out, out_sim, self.metric, raw_grad=raw_grad) # shape: parallel_eq_n,b,H,dim1
                    similarity = similarity.mean([3]) # shape: parallel_eq_n,b,H (remaining mean operation will be done later on)
                    similarity = similarity.sum(dim=1, keepdim=True) # shape: parallel_eq_n,1,H
                    similarities.append(similarity)
                # calculate best similarity for this block
                similarities = torch.cat(similarities, 0) # shape: eq_n,1,H
                batch_similarities.append(similarities)
            batch_similarities = torch.cat(batch_similarities, dim=1).sum(dim=1, keepdim=False) #shape: eq_n,H
            batch_similarities = F.pad(batch_similarities, [0,self.pad_groups_B]).view(self.eq_n,self.n_G_B,self.crb_groups_B).mean(-1) # shape: eq_n, n_G_B
            best_index = torch.argmax(batch_similarities, dim=0, keepdim=False).view(1,1,-1,1,1,1,1,1)
            tmp_B_interval[:,:,:,:,v:v+1,:,h:h+1,:] = torch.gather(B_interval_candidates[:,:,:,:,v:v+1,:,h:h+1,:],dim=0,index=best_index)
            if self.revised:
                B_zeros = torch.cat(B_zeros, dim=0)
                B_zeros = torch.gather(B_zeros, dim=0, index=best_index) 
        self.B_interval = tmp_B_interval.squeeze(0)
        if self.revised:
            self.B_zero = B_zeros.squeeze(0)
        

    # def _random_sample_calib_raw_data(self):
    #     random_ind = torch.randperm(self.calib_size)[:32]
    #     self.raw_input[1] = self.raw_input[1][random_ind]
    #     self.calib_size = 32
    
    def _random_sample_calib_raw_data(self):
        if isinstance(self.raw_input[1], dict): 
            # Timewise의 경우 A는 원래 dict고, B의 calibration을 위한 데이터를 따로 32개 만듦(raw input, out, grad 모두)
            self.calib_size = 32    
            random_ind = torch.randperm(len(self.raw_input[1])*len(self.raw_input[1][0]))[:self.calib_size]
            time_groups = []
            self.raw_input_1 = dict()
            self.raw_input_1[0] = []
            self.raw_input_1[1] = []
            self.raw_out_1 = []
            self.raw_grad_1 = []
            for ind in random_ind:
                index = (ind // self.timegroup_num).item()
                time_group = (ind % self.timegroup_num).item()
                time_groups.append(time_group)
                self.raw_input_1[0].append(self.raw_input[0][time_group][index])
                self.raw_input_1[1].append(self.raw_input[1][time_group][index])
                self.raw_out_1.append(self.raw_out[time_group][index])
                self.raw_grad_1.append(self.raw_grad[time_group][index])
            self.time_groups = time_groups
            self.raw_input_1[0] = torch.stack(self.raw_input_1[0])
            self.raw_input_1[1] = torch.stack(self.raw_input_1[1])
            self.raw_out_1 = torch.stack(self.raw_out_1)
            self.raw_grad_1 = torch.stack(self.raw_grad_1)
                
        elif isinstance(self.raw_input[1], list):
            # Timewise의 경우 A는 원래 dict고, B의 calibration을 위한 데이터를 따로 32개 만듦(raw input, out, grad 모두)
            self.calib_size = 32
            inter_batch_num = self.calib_size // self.raw_input[1][0].shape[0] # 32 // batch_size(4) = 8
            random_ind = torch.randperm(len(self.raw_input[1]))[:inter_batch_num]
            self.raw_input[1] = torch.cat([self.raw_input[1][i] for i in random_ind],dim=0)
            self.raw_out = torch.cat([self.raw_out[i] for i in random_ind],dim=0)
            self.raw_grad = torch.cat([self.raw_grad[i] for i in random_ind],dim=0)
        else:
            self.calib_size = 32
            random_ind = torch.randperm(len(self.raw_input))[:self.calib_size]
            self.raw_input[1] = torch.stack([self.raw_input[1][i] for i in random_ind])
            self.raw_out = torch.stack([self.raw_out[i] for i in random_ind])
            self.raw_grad = torch.stack([self.raw_grad[i] for i in random_ind])            
        
    def calibration_step2(self,device,group_num):
        self.device = device
        self.timegroup_num = group_num
        self.split = dict()
        self._initialize_calib_parameters()
        self._random_sample_calib_raw_data()
        self._initialize_intervals()
        # self._initialize_timegroup()
        A_split_candidates = torch.tensor([2**(-i) for i in range(20)]).to(self.device)
        B_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.device).view(-1,1,1,1,1,1,1,1) * self.B_interval.unsqueeze(0)
        for e in range(self.search_round):
            # search for best A interval
            for i in range(self.timegroup_num):
                self._search_best_A_interval(A_split_candidates, timegroup_ind = i)
            # search for best B interval
            self._search_best_B_interval(B_interval_candidates)
        self.calibrated = True  
        print(f"best split: {self.split}")
        from YH_weight_distribution_plot import plot_splitpoint_pergroup
        plot_splitpoint_pergroup(self.split, 2**(self.A_bit-1))
        del self.raw_input, self.raw_out, self.raw_grad
    
    def quant_forward(self, A, B, timegroup_ind):
        # assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        assert self.A_interval[0] is not None,f"You should define self.A_interval before run quant_forward for {self}"
        A_sim=self.quant_input_A_forward(A, timegroup_ind = timegroup_ind)
        B_sim=self.quant_input_B(B)
        self.A_quant = A_sim
        self.B_quant = B_sim
        out=A_sim@B_sim
        return out
    
    def forward(self, A,B,timegroup_ind=None):
        if self.mode=='raw':
            # if timegroup_ind != None:
            #     if A[0].dim()==2:
            #         max = abs(A[0]).max(dim=1)[0]
            #         min = abs(A[0]).min(dim=1)[0]
            #     if A[0].dim()==3:
            #         for i in range(A[0].shape[0]):
            #             max = abs(A[0]).max(dim=1)[0]
            #             weight_plot(max, f"SoftmaxSoS_raw_A_max_{i}")
            #         max = abs(A[0]).max(dim=1)[0].max(dim=1)[0]
            #         min = abs(A[0]).min(dim=1)[0]
            #     weight_plot(max, "SoftmaxSoS_raw_A")
            out=A @ B
        elif self.mode=="quant_forward":
            out=self.quant_forward(A,B,timegroup_ind=timegroup_ind)
        elif self.mode=="calibration_step1":
            out=self.calibration_step1(A,B)
        elif self.mode=="calibration_step2":
            out=self.calibration_step2(A,B)
        else:
            raise NotImplementedError
        return out