import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional,Tuple

rank = dist.get_rank()
world_size = dist.get_world_size()

def split_weight_col(full_weight : torch.Tensor):
    #full_weight : (in_dim,out_dim)
    chunk_size = full_weight.shape[1] // world_size
    weight_slice = full_weight.chunk(world_size,dim = 1)[rank]
    return weight_slice.clone().detach()    #clone()避免共享 detach()脱离计算图 避免依赖问题


def split_weight_row(full_weight : torch.Tensor):
    #full_weight : (in_dim,out_dim)
    chunk_size = full_weight.shape[0] // world_size
    weight_slice = full_weight.chunk(world_size,dim = 0)[rank]
    return weight_slice.clone().detach()


def split_bias_col(full_bias : torch.Tensor):
    #列切分后维度为 (in_dim,out_dim // world_size) 因此为对其维度 bias也要切分
    chunk_size = full_bias.shape[0] // world_size
    bias_slice = full_bias.chunk(world_size,dim = 0)[rank]
    return bias_slice.clone().detach()


class ColumnParallelLinear(nn.Module):

    def __init__(self,in_features,out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_partition = out_features // world_size

        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition,self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))

    def forward(self,x):
        output_parallel = F.linear(x,self.weight,self.bias)
        return output_parallel



class RowParallelLinear(nn.Module):

    def __init__(self,in_features,out_features):
        super().__init__()
        self.in_features_per_partition = in_features // world_size
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(self.out_features,self.in_features_per_partition))
        self.bias = nn.Parameter(torch.empty(self.out_features))
    
    def forward(self,x_parallel):
        partial_output = F.linear(x_parallel,self.weight)
        dist.all_reduce(partial_output,op = dist.ReduceOp.SUM)
        output = partial_output + self.bias