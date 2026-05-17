import sys
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
from collections import OrderedDict, defaultdict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from data import ImageFolder
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

if __name__ == '__main__':
    ### debug
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)

    # dist.init_process_group("nccl") ### 
    assert 256 % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = 0 * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    data,target=[],[]
    path="/root/autodl-tmp/extend/"
    for folder in os.listdir(path):
        if folder==".ipynb_checkpoints":
            continue
        path_I=os.path.join(path,folder)
        for Images in os.listdir(path_I):
            if Images==".ipynb_checkpoints":
                continue
            path_i=os.path.join(path_I,Images)
            image=Image.open(path_i)
            t_image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().cuda()
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(t_image.unsqueeze(0)).latent_dist.sample().mul_(0.18215)
            data.append(x.detach().cpu())
            target.append(int(folder))
    torch.save([data,target],os.path.join("/root/autodl-tmp/",f'extend_data.pt'))
    
    