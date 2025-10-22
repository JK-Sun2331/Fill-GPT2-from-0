import torch.distributed as dist
import torch
import os

#CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 your_inference_script.py

def setup_distributed():

    if not dist.is_initialized():
        local_rank = int(os.environ.get('LOCAL_RANK'))
        torch.cuda.set_device(local_rank)   #设置当前进程使用的GPU

        dist.init_process_group(backend = 'nccl')   #初始化进程组
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    
    return rank,world_size

rank,world_size = setup_distributed()
print(rank,world_size)