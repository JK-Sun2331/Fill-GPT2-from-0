import torch
import torch.nn as nn
from safetensors.torch import load_file

def load_weight():
    model_path = "/data1/hfhub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors" 

    state_dict = load_file(model_path)
    
    weight = state_dict["h.0.attn.c_attn.weight"]

    print(weight.size())

load_weight()