import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class ColumnParallelLinear(nn.Module):

    def __init__(self,in_features,out_features,tp_rank,tp_size):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_partition = out_features // tp_size

        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition,self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))

    def forward(self,x):
        output_parallel = F.linear(x,self.weight,self.bias)
        return output_parallel



class RowParallelLinear(nn.Module):

    def __init__(self,in_features,out_features,tp_rank,tp_size):
        super().__init__()
        self.in_features_per_partition = in_features // tp_size
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(self.out_features,self.in_features_per_partition))
        self.bias = nn.Parameter(torch.empty(self.out_features))
    
    def forward(self,x_parallel):
        partial_output = F.linear(x_parallel,self.weight)
        dist.all_reduce(partial_output,op = dist.ReduceOp.SUM)
        output = partial_output + self.bias
        return output