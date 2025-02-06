from numpy import isin
import torch
from quant_layers.conv import MinMaxQuantConv2d
from quant_layers.linear import MinMaxQuantLinear, PTQSLQuantLinear, PostGeluPTQSLBatchingQuantLinear, PostNormPTQSLBatchingQuantLinear, PTQSLBatchingQuantLinear
from quant_layers.matmul import MinMaxQuantMatMul, PTQSLQuantMatMul, SoSPTQSLBatchingQuantMatMul, LISPTQSLBatchingQuantMatMul, SoLPTQSLBatchingQuantMatMul, TimewiseSoSPTQSLBatchingQuantMatMul
import torch.nn.functional as F
from tqdm import tqdm
from diffusion import create_diffusion
from YH_weight_distribution_plot import plot_channel_minmax, weight_plot
import numpy as np
from utils.global_var import get_model_input, hook, get_hook_data, get_group_ind

class QuantCalibrator():
    """
    Modularization of quant calib.

    Notice: 
    all quant modules has method "calibration_step1" that should only store raw inputs and outputs
    all quant modules has method "calibration_step2" that should only quantize its intervals
    and we assume we could feed in all calibration data in one batch, without backward propagations

    sequential calibration is memory-friendly, while parallel calibration may consume 
    hundreds of GB of memory.
    """
    def __init__(self, net, wrapped_modules, calib_loader, sequential=True):
        self.net = net
        self.wrapped_modules = wrapped_modules
        self.calib_loader = calib_loader
        self.sequential = sequential
        self.calibrated = False
    
    def sequential_quant_calib(self):
        """
        A quick implementation of calibration.
        Assume calibration dataset could be fed at once.
        """
        # run calibration
        n_calibration_steps=2
        for step in range(n_calibration_steps):
            print(f"Start calibration step={step+1}")
            for name,module in self.wrapped_modules.items():
                # corner cases for calibrated modules
                if hasattr(module, "calibrated"):
                    if step == 1:
                        module.mode = "raw"
                    elif step == 2:
                        module.mode = "quant_forward"
                else:
                    module.mode=f'calibration_step{step+1}'
            with torch.no_grad():
                for inp,target in self.calib_loader:
                    inp=inp.cuda()
                    self.net(inp)
        
        # finish calibration
        for name,module in self.wrapped_modules.items():
            module.mode='quant_forward'
        torch.cuda.empty_cache() # memory footprint cleanup
        print("sequential calibration finished")
    
    def parallel_quant_calib(self):
        """
        A quick implementation of parallel quant calib
        Assume calibration dataset could be fed at once, and memory could hold all raw inputs/outs
        """
        # calibration step1: collect raw data
        print(f"Start calibration step=1")
        for name,module in self.wrapped_modules.items():
            # corner cases for calibrated modules
            if hasattr(module, "calibrated"):
                module.mode = "raw"
            else:
                module.mode=f'calibration_step1'
        with torch.no_grad():
            for inp,target in self.calib_loader:
                inp=inp.cuda()
                self.net(inp)
        # calibration step2: each module run calibration with collected raw data
        for name,module in self.wrapped_modules.items():
            if hasattr(module, "calibrated"):
                continue
            else:
                module.mode=f"calibration_step2"
                with torch.no_grad():
                    if isinstance(module, MinMaxQuantLinear):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, MinMaxQuantConv2d):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, MinMaxQuantMatMul):
                        module.forward(module.raw_input[0].cuda(), module.raw_input[1].cuda())
                    torch.cuda.empty_cache()
                
        # finish calibration
        for name,module in self.wrapped_modules.items():
            module.mode='quant_forward'
        torch.cuda.empty_cache() # memory footprint cleanup
        print("calibration finished")
    
    def quant_calib(self):
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")
        if self.sequential:
            self.sequential_quant_calib()
        else:
            self.parallel_quant_calib()
        self.calibrated = True

    def batching_quant_calib(self):        
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")

        print("start calibration")

        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items(), desc="Brecq")
        for name, module in q:
            q.set_postfix_str(name)

            # add fp and bp hooks to current modules, which bypass calibration step 1
            # precedent modules are using quant forward
            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_hook(matmul_forward_hook))
            
            # feed in calibration data, and store the data
            for inp, target in self.calib_loader:
                for batch_st in range(0,self.calib_loader.batch_size,self.batch_size):
                    self.net.zero_grad()
                    inp_ = inp[batch_st:batch_st+self.batch_size].cuda()
                    self.net(inp_)
                del inp, target
                torch.cuda.empty_cache()
            
            # replace cached raw_inputs, raw_outs
            if isinstance(module, MinMaxQuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantConv2d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantMatMul):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            for hook in hooks:
                hook.remove()

            # run calibration step2
            with torch.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2()
                torch.cuda.empty_cache()
            
            # finishing up current module calibration
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        
        print("calibration finished")

def grad_hook(module, grad_input, grad_output):
    if isinstance(module, TimewiseSoSPTQSLBatchingQuantMatMul):
        if module.raw_grad is None:
            module.raw_grad = dict()
        timegroup_key = get_model_input() 
        if not timegroup_key in module.raw_grad:
            module.raw_grad[timegroup_key] = []
        module.raw_grad[timegroup_key].append(grad_output[0].cpu().detach()) # grad도 dict로 수정 0905
    else:
        if module.raw_grad is None:
            module.raw_grad = []
        module.raw_grad.append(grad_output[0].cpu().detach())   # that's a tuple!

def linear_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())
    # if isinstance(module, PTQSLBatchingQuantLinear) and module.out_features == 3456:
    # if isinstance(module, PostGeluPTQSLBatchingQuantLinear):
    #     model_input = get_model_input()
    #     for t_ind in range(len(model_input)):
    #         if module.raw_input[0][t_ind] !=[]:
    #             t = model_input[t_ind]
    #             hook(t,module.raw_input[0], t_ind)
            
def conv2d_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())

def matmul_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = [[],[]]
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input[0].append(input[0].cpu().detach())
    module.raw_input[1].append(input[1].cpu().detach())
    module.raw_out.append(output.cpu().detach())
    # if isinstance(module, SoSPTQSLBatchingQuantMatMul):
    #     model_input = get_model_input()
    #     if model_input.dim() == 2:
    #         t_ind = 0
    #         if module.raw_input[0][0][t_ind] !=[]:
    #             t = model_input[t_ind]
    #             hook(t,module.raw_input[0][0], t_ind)
    #     else:
    #         for t_ind in range(len(model_input)):
    #             if module.raw_input[0][0][t_ind] !=[]:
    #                 t = model_input[t_ind]
    #                 hook(t,module.raw_input[0][0], t_ind)

def matmul_forward_hook_timegroup(module, input, output):
    if module.raw_input is None:
        module.raw_input = [dict(),dict()]
    if module.raw_out is None:
        module.raw_out = dict()
    # timegroup_key = module.timestep[-1]
    timegroup_key = get_model_input() # 여기 하고 있었음 0813
    if not timegroup_key in module.raw_input[0]:
        module.raw_input[0][timegroup_key] = []
        module.raw_input[1][timegroup_key] = []
        module.raw_out[timegroup_key] = []
    module.raw_input[0][timegroup_key].append(input[0].cpu().detach())
    module.raw_input[1][timegroup_key].append(input[1].cpu().detach()) # B도 dict로 수정 0905
    module.raw_out[timegroup_key].append(output.cpu().detach()) # out도 dict로 수정 0905
    # module.raw_input[1].append(input[1].cpu().detach())
    # module.raw_out.append(output.cpu().detach())

class HessianQuantCalibrator(QuantCalibrator):
    """
    Modularization of hessian_quant_calib

    Hessian metric needs gradients of layer outputs to weigh the loss,
    which calls for back propagation in calibration, both sequentially
    and parallelly. Despite the complexity of bp, hessian quant calibrator
    is compatible with other non-gradient quantization metrics.
    """
    def __init__(self, net, wrapped_modules, calib_loader, sequential=False, batch_size=1):
        super().__init__(net, wrapped_modules, calib_loader, sequential=sequential)
        self.batch_size = batch_size

    def quant_calib(self):
        """
        An implementation of original hessian calibration.
        """

        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")

        print("start hessian calibration")

        # get raw_pred as target distribution 
        with torch.no_grad():
            for inp, _ in self.calib_loader:
                raw_pred = self.net(inp.cuda())
                raw_pred_softmax = F.softmax(raw_pred, dim=-1).detach()
            torch.cuda.empty_cache()

        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items(), desc="Brecq")
        for name, module in q:
            q.set_postfix_str(name)

            # add fp and bp hooks to current modules, which bypass calibration step 1
            # precedent modules are using quant forward
            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_hook(matmul_forward_hook))
            if hasattr(module, "metric") and module.metric == "hessian":
                hooks.append(module.register_backward_hook(grad_hook))
            
            # feed in calibration data, and store the data
            for inp, target in self.calib_loader:
                for batch_st in range(0,self.calib_loader.batch_size,self.batch_size):
                    self.net.zero_grad()
                    inp_ = inp[batch_st:batch_st+self.batch_size].cuda()
                    pred = self.net(inp_)
                    loss = F.kl_div(F.log_softmax(pred, dim=-1), raw_pred_softmax[batch_st:batch_st+self.batch_size], reduction="batchmean")
                    loss.backward()
                del inp, target, pred, loss
                torch.cuda.empty_cache()
            
            # replace cached raw_inputs, raw_outs
            if isinstance(module, MinMaxQuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantConv2d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantMatMul):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if hasattr(module, "metric") and module.metric == "hessian":
                module.raw_grad = torch.cat(module.raw_grad, dim=0)
            for hook in hooks:
                hook.remove()

            # run calibration step2
            with torch.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2(module.raw_input[0].cuda(), module.raw_input[1].cuda())
                torch.cuda.empty_cache()
            
            # finishing up current module calibration
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        
        print("hessian calibration finished")

    def batching_quant_calib(self,args):
        last_norm = None
        last_gelu= None
        calib_layers=[]
        current_metric = ''
        cnt=0
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
            module.sym = True
            if module.metric != current_metric:
                current_metric = module.metric
                print("Check current metric:", current_metric)
            # if isinstance(module, PostGeluPTQSLBatchingQuantLinear):
            #     last_gelu = module
            if isinstance(module, SoSPTQSLBatchingQuantMatMul):
                last_sos = module
                cnt+=1
            # if isinstance(module, SoLPTQSLBatchingQuantMatMul):
            #     last_sol = module
            #     # cnt+=1

        # for m in self.net.modules():
        #     if isinstance(m, torch.nn.LayerNorm):
        #         last_norm = m
        
        # def hook_minmax(module, input, output):
        #     inp = input[0]
        #     module.mins_in = inp.min(dim=0)[0].min(dim=0)[0].detach().cpu().numpy()
        #     module.maxs_in = inp.max(dim=0)[0].min(dim=0)[0].detach().cpu().numpy()
        #     module.mins_out = output.min(dim=0)[0].min(dim=0)[0].detach().cpu().numpy()
        #     module.maxs_out = output.max(dim=0)[0].min(dim=0)[0].detach().cpu().numpy()
        # def hook_dist(module, input, output):
        #     inp = input[0]
        #     module.dist = inp[0].detach().cpu().numpy()
        # def hook_dist_matmul(module, input, output):
        #     inpA = input[0]
        #     inpB = input[1]
        #     module.distA = inpA[0].detach().cpu().numpy()
        #     module.distB = inpB[0].detach().cpu().numpy()
        
        # plot = False
        # if plot:
        #     hook_1 = last_norm.register_forward_hook(hook_minmax)
        #     hook_2 = last_gelu.register_forward_hook(hook_dist)
        #     # hook_3 = last_sos.register_forward_hook(hook_dist_matmul)
                
        print(f"prepare parallel calibration for {calib_layers}")
        print("start hessian calibration")
                
        # get raw_pred as target distribution 
        # diffusion_steps = 1000
        # time_steps = 250
        # timestep_map = torch.linspace(0, diffusion_steps - 1, time_steps, device='cuda:0')
        # map_tensor = torch.tensor(timestep_map, device=args.device, dtype=torch.int64)
        map = torch.linspace(0, 999, 250).round().long().to(args.device) # 0830
        with torch.no_grad():
            for inp in self.calib_loader:
                model_kwargs = dict(y=inp[2].to(args.device), cfg_scale=args.cfg_scale)
                # new_ts = map_tensor[inp[1]]
                if inp[0].dim() == 5:
                    inp[0] = inp[0].reshape(64,4,32,32)
                    inp[1] = inp[1].reshape(-1)
                    model_kwargs['y'] = model_kwargs['y'].reshape(-1)
                
                if args.extraction == "random" or args.extraction == "random timewise":
                    inp[1] = map[inp[1]] # 0830
                # raw_pred = self.net(inp[0].to(args.device), inp[1].to(args.device), model_kwargs['y'])
                raw_pred = self.net.forward_with_cfg(inp[0].to(args.device), inp[1].to(args.device), **model_kwargs)
                raw_pred_softmax = F.softmax(raw_pred, dim=-1).detach()
            torch.cuda.empty_cache()
        
        # if plot:
        #     plot_channel_minmax(last_norm.mins_in, last_norm.maxs_in, "last_norm_in")
        #     plot_channel_minmax(last_norm.mins_out, last_norm.maxs_out, "last_norm_out")
        #     weight_plot(last_gelu.dist, "last_gelu")
        #     # weight_plot(last_sos.distA, "last_sos_raw_A_log", log = True)
        #     # weight_plot(last_sos.distA, "last_sos_raw_A", bins = 256)
        #     # weight_plot(last_sos.distB, "last_sos_raw_B_log", log = True)
        #     # weight_plot(last_sos.distB, "last_sos_raw_B", bins = 256)
        #     hook_1.remove()
        #     hook_2.remove()
        #     # hook_3.remove()
        cnt2 = 1

        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items(), desc="Hessian")
        for name, module in q:
            if not isinstance(module, SoSPTQSLBatchingQuantMatMul):
                continue
            # if not (isinstance(module, PTQSLBatchingQuantLinear) and module.out_features == 3456):
            #     continue
            # if not isinstance(module, PostNormPTQSLBatchingQuantLinear):
            #     continue
            # elif cnt2 != cnt:
            #     cnt2+=1
            #     continue
            q.set_postfix_str(name)

            # add fp and bp hooks to current modules, which bypass calibration step 1
            # precedent modules are using quant forward
            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_hook(matmul_forward_hook))
            if hasattr(module, "metric"):
                hooks.append(module.register_backward_hook(grad_hook))
            
            # feed in calibration data, and store the data                
            for input_datas in self.calib_loader:
                for batch_st in range(0,self.calib_loader.batch_size,self.batch_size):
                    x = input_datas[0][batch_st:batch_st+self.batch_size].to(args.device)
                    t = input_datas[1][batch_st:batch_st+self.batch_size].to(args.device)
                    y = input_datas[2][batch_st:batch_st+self.batch_size].to(args.device)
                    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                    self.net.zero_grad()
                    
                    from utils.global_var import set_model_input                    
                    set_model_input(t)
                    if args.extraction == "random" or args.extraction == "random timewise":
                        t = map[t]
                    pred = self.net.forward_with_cfg(x, t, **model_kwargs)
                    # pred = self.net(x, t, model_kwargs['y'])
                    loss = F.kl_div(F.log_softmax(pred, dim=-1), raw_pred_softmax[batch_st:batch_st+self.batch_size], reduction="batchmean")
                    loss.backward()
                del x, t, y, input_datas, pred, loss
                torch.cuda.empty_cache()
            
            # replace cached raw_inputs, raw_outs
            if isinstance(module, MinMaxQuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantConv2d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantMatMul):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if hasattr(module, "metric"):
                module.raw_grad = torch.cat(module.raw_grad, dim=0)
            for hook in hooks:
                hook.remove()

            # run calibration step2
            with torch.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2(args.device)
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2(args.device)
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2(args.device)
                torch.cuda.empty_cache()
            
            # finishing up current module calibration
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"
                
            # if plot:
            #     module.mode = "quant_forward"
            #     with torch.no_grad():
            #         for inp in self.calib_loader:
            #             model_kwargs = dict(y=inp[2].to(args.device), cfg_scale=args.cfg_scale)
            #             raw_pred = self.net.forward_with_cfg(inp[0].to(args.device), inp[1].to(args.device), **model_kwargs)
            #             raw_pred_softmax = F.softmax(raw_pred, dim=-1).detach()
            #         torch.cuda.empty_cache()
            #     # weight_plot(last_sos.A_quant, "last_sos_quantized_A_log", log=True)
            #     # weight_plot(last_sos.A_quant, "last_sos_quantized_A", bins = 256)
            #     # weight_plot(last_sol.A_quant, "last_sol_quantized_A_log", log=True)
            #     # weight_plot(last_sol.A_quant, "last_sol_quantized_A", bins = 256)
            
            # """
            # Time-step activation plot (Linear in Attention)
            # """
            from YH_weight_distribution_plot import activation_box_plot
            hook_data = get_hook_data()
            processed_data = []
            labels = []
            sorted_items = sorted(hook_data.items(), key=lambda x: int(x[0]))
            for idx, (label, data_list) in enumerate(sorted_items):
                if data_list[0].dim()==2:
                    all_data = abs(data_list[0]).max(dim=1)[0]
                if data_list[0].dim()==3:
                    all_data = abs(data_list[0]).max(dim=1)[0].max(dim=1)[0]
                # all_data = np.concatenate([arr.flatten() for arr in data_list])
                processed_data.append(all_data)
                labels.append(f"{label}")
            activation_box_plot(processed_data, labels, "SoftmaxSoS_channel_calib")
            break

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        
        print("hessian calibration finished")
    
    def batching_quant_calib_timestep(self,args):
        last_norm = None
        last_gelu= None
        calib_layers=[]
        current_metric = ''
        cnt=0
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
            if module.metric != current_metric:
                current_metric = module.metric
                print("Check current metric:", current_metric)
            if isinstance(module, PostGeluPTQSLBatchingQuantLinear):
                last_gelu = module
            if isinstance(module, SoSPTQSLBatchingQuantMatMul):
                last_sos = module
            if isinstance(module, SoLPTQSLBatchingQuantMatMul):
                last_sol = module
                # cnt+=1
            if isinstance(module, TimewiseSoSPTQSLBatchingQuantMatMul):
                last_sostime = module
                cnt += 1

        for m in self.net.modules():
            if isinstance(m, torch.nn.LayerNorm):
                last_norm = m
        
        print(f"prepare parallel calibration for {calib_layers}")
        print("start hessian calibration")
        
        # get raw_pred as target distribution 
        # diffusion_steps = 1000
        # time_steps = 250
        # timestep_map = torch.linspace(0, diffusion_steps - 1, time_steps, device='cuda:0')
        # map_tensor = torch.tensor(timestep_map, device=args.device, dtype=torch.int64)
        raw_pred_softmax = []
        map = torch.linspace(0, 999, 250).round().long().to(args.device) # 0830
        with torch.no_grad():
            for inp in self.calib_loader:
                model_kwargs = dict(y=inp[2].to(args.device), cfg_scale=args.cfg_scale)                
                if args.extraction == "random" or args.extraction == "random timewise":
                    inp[1] = map[inp[1]] # 0830
                raw_pred = self.net.forward_with_cfg(inp[0].to(args.device), inp[1].to(args.device), **model_kwargs)
                raw_pred_softmax.append(F.softmax(raw_pred, dim=-1).detach())
            torch.cuda.empty_cache()
        cnt2 = 1
        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items(), desc="Hessian")
        for name, module in q:
            # if name != 'blocks.0.mlp.fc1':
            #     continue
            # if not isinstance(module, MinMaxQuantMatMul):
            #     continue
            # elif cnt2 != cnt:
            #     cnt2 += 1
            #     continue
            q.set_postfix_str(name)
    
            # add fp and bp hooks to current modules, which bypass calibration step 1
            # precedent modules are using quant forward
            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                if isinstance(module, TimewiseSoSPTQSLBatchingQuantMatMul):
                    hooks.append(module.register_forward_hook(matmul_forward_hook_timegroup))
                else:
                    hooks.append(module.register_forward_hook(matmul_forward_hook))
            if hasattr(module, "metric"):
                hooks.append(module.register_backward_hook(grad_hook))
            
            # feed in calibration data, and store the data                
            i=0
            for input_datas in self.calib_loader:
                for batch_st in range(0,self.calib_loader.batch_size,self.batch_size):
                    x = input_datas[0][batch_st:batch_st+self.batch_size].to(args.device)
                    t = input_datas[1][batch_st:batch_st+self.batch_size].to(args.device)
                    y = input_datas[2][batch_st:batch_st+self.batch_size].to(args.device)
                    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                    self.net.zero_grad()
                    
                    if args.extraction == "random" or args.extraction == "random timewise":
                        t = map[t]
                        
                    from utils.global_var import set_model_input, get_group_ind
                    group_ind = get_group_ind(t, option = "calib", total_timestep = args.num_sampling_steps, group_num = args.group_num)
                    set_model_input(group_ind)
                        
                    pred = self.net.forward_with_cfg(x, t, **model_kwargs)
                    loss = F.kl_div(F.log_softmax(pred, dim=-1), raw_pred_softmax[i][batch_st:batch_st+self.batch_size], reduction="batchmean")
                    loss.backward() 
                i += 1
                del x, t, y, pred, loss
            del input_datas
            torch.cuda.empty_cache()
            
            if args.extraction == "timewise" or args.extraction == "random timewise":
                if isinstance(module, MinMaxQuantLinear):
                    module.raw_input = torch.cat(module.raw_input, dim=0)
                    module.raw_out = torch.cat(module.raw_out, dim=0)
                    if hasattr(module, "metric"):
                        module.raw_grad = torch.cat(module.raw_grad, dim=0)
                if isinstance(module, MinMaxQuantMatMul):
                    if isinstance(module, TimewiseSoSPTQSLBatchingQuantMatMul):
                        # module.raw_input[1] = torch.cat(module.raw_input[1], dim=0) # B
                        for group_key, group_list in module.raw_input[0].items(): # A (after Softmax)
                            module.raw_input[0][group_key] = torch.cat(group_list, dim=0)
                            module.raw_input[1][group_key] = torch.cat(module.raw_input[1][group_key], dim=0)
                            module.raw_out[group_key]      = torch.cat(module.raw_out[group_key], dim=0)
                            module.raw_grad[group_key]     = torch.cat(module.raw_grad[group_key], dim=0)
                    else:
                        # module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                        # module.raw_out = torch.cat(module.raw_out, dim=0)
                        pass
            else:
                # replace cached raw_inputs, raw_outs
                if isinstance(module, MinMaxQuantLinear):
                    module.raw_input = torch.cat(module.raw_input, dim=0)
                    module.raw_out = torch.cat(module.raw_out, dim=0)
                if isinstance(module, MinMaxQuantConv2d):
                    module.raw_input = torch.cat(module.raw_input, dim=0)   
                    module.raw_out = torch.cat(module.raw_out, dim=0)
                if isinstance(module, MinMaxQuantMatMul):
                    module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                    module.raw_out = torch.cat(module.raw_out, dim=0)
                if hasattr(module, "metric"):
                    module.raw_grad = torch.cat(module.raw_grad, dim=0)
            for hook in hooks:
                hook.remove()

            # run calibration step2
            with torch.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2(args.device)
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2(args.device)
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2(args.device, args.group_num)
                torch.cuda.empty_cache()
            
            # finishing up current module calibration
            if self.sequential: 
                module.mode = "quant_forward"
            else:
                module.mode = "raw"
                
            """
            Time-step activation plot (Linear in Attention)
            """
            from YH_weight_distribution_plot import activation_box_plot
            hook_data = get_hook_data()
            processed_data = []
            labels = []
            sorted_items = sorted(hook_data.items(), key=lambda x: int(x[0]))
            for idx, (label, data_list) in enumerate(sorted_items):
                if data_list[0].dim()==2:
                    all_data = abs(data_list[0]).max(dim=1)[0]
                if data_list[0].dim()==3:
                    all_data = abs(data_list[0]).max(dim=1)[0].max(dim=1)[0]
                # all_data = np.concatenate([arr.flatten() for arr in data_list])
                processed_data.append(all_data)
                labels.append(f"{label}")
            activation_box_plot(processed_data, labels, "SoftmaxSoS_channel_calib")
            break

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        
        print("hessian calibration finished")
    
        