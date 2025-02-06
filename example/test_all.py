from timm.models.layers import config
from torch.nn.modules import module
from quant_layers.conv import MinMaxQuantConv2d
from quant_layers.linear import MinMaxQuantLinear, PTQSLQuantLinear
from quant_layers.matmul import MinMaxQuantMatMul, PTQSLQuantMatMul
import matplotlib.pyplot as plt
import utils.net_wrap as net_wrap
from utils.net_wrap import wrap_certain_modules_in_net
from utils.nets import DiT_models
from tqdm import tqdm
import torch.nn.functional as F
import pickle as pkl
from itertools import product
import types
from utils.quant_calib import HessianQuantCalibrator, QuantCalibrator
from utils.models import get_net
import time
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from quant_dataset import DiffusionInputDataset, calib_loader
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import datetime
import torch
from importlib import reload,import_module
import argparse
import utils.integer as integer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'

def test_all(name, args, cfg_modifier=lambda x: x, calib_size=32, config_name="TQ-DiT"):
    args.device = "cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu"
    args.cfg_scale = 1.5
    args.image_size = 256
    quant_cfg = init_config(config_name)
    quant_cfg = cfg_modifier(quant_cfg) 
    batch_size = 4
    group_num = 10
    latent_size = args.image_size // 8
    args.group_num = group_num
    net = get_net(name, args.image_size, args.num_classes, args.device, args.ckpt)
    if name == "DiT-XL/2":
        path = "DiT-XL-2"
    wrapped_modules=net_wrap.wrap_modules_in_net(net,quant_cfg)

    # for DiT
    extraction = "random timewise" # dataset extraction method ("random" or "uniform class" or "timewise" or "random timewise")
    args.extraction = extraction
    dataset_path = "imagenet_input_250steps.pth" 
    dataset = DiffusionInputDataset(dataset_path, calib_size = calib_size, extraction=extraction, group_num = group_num, timestep = args.num_sampling_steps)
    data_loader = calib_loader(dataset=dataset, num_samples=calib_size, batch_size=calib_size, extraction=extraction, group_num = group_num)
    
    # add timing
    print("Start time:", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    calib_start_time = time.time()
    torch.cuda.reset_peak_memory_stats(device=args.device)
    
    quant_calibrator = HessianQuantCalibrator(net,wrapped_modules,data_loader,sequential=False,batch_size=batch_size)
    if config_name=="TQ-DiT" or extraction == "random timewise":
        quant_calibrator.batching_quant_calib_timestep(args)
    else:
        quant_calibrator.batching_quant_calib(args)
    calib_end_time = time.time()
    max_memory_used = torch.cuda.max_memory_allocated(device=args.device)
    max_memory_used_mb = max_memory_used / (1024 ** 2)

    print(f"Peak GPU Memory Used on {args.device}: {max_memory_used_mb:.2f} MB")
    print("End time:", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    print(f"calibration time: {(calib_end_time-calib_start_time)/60}min \n")
    
    # torch.save(net.state_dict(), 'results/{}_PTQ_W{}A{}.pth'.format(name, cfg_modifier.bit_setting[0],cfg_modifier.bit_setting[1]))
    intervals = dict()
    for m_name, module in wrapped_modules.items():
        if 'matmul' in m_name:
            intervals[m_name+'.A_interval'] = module.A_interval
            intervals[m_name+'.B_interval'] = module.B_interval
            intervals[m_name+'.A_shape'] = module.A_shape
            intervals[m_name+'.B_shape'] = module.B_shape
            if hasattr(module,'A_zero'):
                intervals[m_name+'.A_zero'] = module.A_zero
            if hasattr(module,'B_zero'):
                intervals[m_name+'.B_zero'] = module.B_zero
            if 'matmul2' in m_name:
                if hasattr(module,'split'):
                    intervals[m_name+'.split'] = module.split 
        else:
            intervals[m_name+'.w_interval'] = module.w_interval
            intervals[m_name+'.a_interval'] = module.a_interval
            if hasattr(module,'w_zero'):
                intervals[m_name+'.w_zero'] = module.w_zero
            if hasattr(module,'a_zero'):
                intervals[m_name+'.a_zero'] = module.a_zero
    torch.save(intervals, 'results/{}_intervals_W{}A{}_{}_seed{}_{}step_{}group.pth'.format(path, cfg_modifier.bit_setting[0],cfg_modifier.bit_setting[1],config_name,args.seed,args.num_sampling_steps,args.group_num))

    if args.sample:
        diffusion = create_diffusion(str(args.num_sampling_steps), group_num=args.group_num)
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(args.device)
        outpath = "./quick_samples/{}_intervals_W{}A{}_{}_seed{}_{}step".format(path, cfg_modifier.bit_setting[0],cfg_modifier.bit_setting[1],config_name,args.seed,args.num_sampling_steps)
        if config_name=="TQ-DiT":
            outpath=outpath+f"_{args.group_num}group"
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        
        # Labels to condition the model with (feel free to change):
        class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

        # Create sampling noise:
        n = len(class_labels)
        z = torch.randn(n, 4, latent_size, latent_size, device=args.device)
        y = torch.tensor(class_labels, device=args.device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=args.device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        import torchprofile
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        start_time = time.time()

        # Sample images:
        samples = diffusion.p_sample_loop( # revised YH
            net.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=args.device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample
        
        end_time = time.time()  
        inference_time = end_time - start_time
        current_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        from utils.global_var import FLOP
        flops = FLOP()
        
        # Save and display images:
        torch.cuda.empty_cache()
        save_image(samples, os.path.join(outpath, "sample.png"), nrow=4, normalize=True, value_range=(-1, 1))
        print(f"Saved samples to {outpath}/sample.png")
        print("[Inference time and memory]")
        print(f"Model {config_name} W{cfg_modifier.bit_setting[0]}A{cfg_modifier.bit_setting[1]} - Inference Time: {inference_time:.6f} seconds")
        print(f"Model {config_name} W{cfg_modifier.bit_setting[0]}A{cfg_modifier.bit_setting[1]} - Current Memory: {current_memory} bytes, Peak Memory: {peak_memory} bytes")
        print(f"Model {config_name} W{cfg_modifier.bit_setting[0]}A{cfg_modifier.bit_setting[1]} - Flops: {flops}")

    
class cfg_modifier():
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self,name,value)

    def __call__(self, cfg):
        # bit setting
        cfg.bit = self.bit_setting
        cfg.w_bit = {name: self.bit_setting[0] for name in cfg.conv_fc_name_list}
        cfg.a_bit = {name: self.bit_setting[1] for name in cfg.conv_fc_name_list}
        cfg.A_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}
        cfg.B_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}

        # conv2d configs
        cfg.ptqsl_conv2d_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_conv2d_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_conv2d_kwargs["metric"] = self.metric
        cfg.ptqsl_conv2d_kwargs["init_layerwise"] = False

        # linear configs
        cfg.ptqsl_linear_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_linear_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_linear_kwargs["n_a"] = self.linear_ptq_setting[2]
        cfg.ptqsl_linear_kwargs["metric"] = self.metric
        cfg.ptqsl_linear_kwargs["init_layerwise"] = False

        # matmul configs
        cfg.ptqsl_matmul_kwargs["metric"] = self.metric
        cfg.ptqsl_matmul_kwargs["init_layerwise"] = False

        return cfg
 
def init_config(config_name):
    """initialize the config. Use reload to make sure it's fresh one!"""
    _,_,files =  next(os.walk("./configs"))
    if config_name+".py" in files:
        quant_cfg = import_module(f"configs.{config_name}")
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")
    reload(quant_cfg)
    return quant_cfg
       
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--multiprocess", action='store_true')
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", action="store_true", default=True, help="sample images")
    parser.add_argument("--ckpt", type=str, default='../pretrained_models/DiT-XL-2-256x256.pt')
    args = parser.parse_args()
    return args

if __name__=='__main__':    
    args = parse_args()
    names = ["DiT-XL/2"]
    metrics = ["hessian", "L2_norm", "cosine"]
    linear_ptq_settings = [(1,1,1)] # n_V, n_H, n_a
    calib_sizes = [32]
    bit_settings = [(8,8),(6,6)] # weight, activation
    # config_names = ["TQ-DiT", "HessianPTQ", "BasePTQ","MSEPTQ"]
    config_names = ["TQ-DiT"]

    cfg_list = []
    for name, metric, linear_ptq_setting, calib_size, bit_setting, config_name in product(names, metrics, linear_ptq_settings, calib_sizes, bit_settings, config_names):
        if config_name == "BasePTQ":
            metric = "hessian"
        elif config_name == "MSEPTQ":
            metric = "L2_norm"
        else 
            metric = "hessian"
            
        cfg_list.append({
            "name": name,
            "cfg_modifier":cfg_modifier(linear_ptq_setting=linear_ptq_setting, metric=metric, bit_setting=bit_setting),
            "calib_size":calib_size,
            "config_name": config_name,
            "args": args
        })

    for cfg in cfg_list:
        test_all(**cfg)
        break
