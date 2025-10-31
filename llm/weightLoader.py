import os
from safetensors.torch import load_file
import torch


def _load_weight(model,model_path,config,tp_rank,tp_size):
        weight_file = os.path.join(model_path,"model.safetensors")
        assert weight_file,"There is no weight_file"

        #加载权重
        state_dict = load_file(weight_file,device=tp_rank)

        #权重名称映射
        weight_mapping = {
            #嵌入权重
            "position_embedding_table.weight" : "wpe.weight",   #(max_seq_len,hidden_size)
            "token_embedding_table.weight" : "wte.weight",      #(vocab_size,hidden_size)
            #最终归一化层权重
            "ln_final.weight" : "ln_f.weight",
            "ln_final.bias" : "ln_f.bias",
            #LMhead权重
            "lm_head.weight" : "wte.weight"         #tie权重
        }

        #Attention块映射
        for i in range(config.n_layer):
            #layernorm
            weight_mapping[f"blocks.{i}.ln1.weight"] = f"h.{i}.ln_1.weight"
            weight_mapping[f"blocks.{i}.ln1.bias"] = f"h.{i}.ln_1.bias"
            weight_mapping[f"blocks.{i}.ln2.weight"] = f"h.{i}.ln_2.weight"
            weight_mapping[f"blocks.{i}.ln2.bias"] = f"h.{i}.ln_2.bias"
            #attention
            weight_mapping[f"blocks.{i}.att.c_attn.weight"] = f"h.{i}.attn.c_attn.weight"
            weight_mapping[f"blocks.{i}.att.c_attn.bias"] = f"h.{i}.attn.c_attn.bias"
            
            weight_mapping[f"blocks.{i}.att.proj.weight"] = f"h.{i}.attn.c_proj.weight"
            weight_mapping[f"blocks.{i}.att.proj.bias"] = f"h.{i}.attn.c_proj.bias"
            #mlp
            weight_mapping[f"blocks.{i}.ffn.net1.weight"] = f"h.{i}.mlp.c_fc.weight"
            weight_mapping[f"blocks.{i}.ffn.net1.bias"] = f"h.{i}.mlp.c_fc.bias"
            weight_mapping[f"blocks.{i}.ffn.net2.weight"] = f"h.{i}.mlp.c_proj.weight"
            weight_mapping[f"blocks.{i}.ffn.net2.bias"] = f"h.{i}.mlp.c_proj.bias"
        
        new_state_dict = {}

        for my_name,hf_name in weight_mapping.items():
            #因为用linear(input,output)初始化的权重 其权重形状是(output,input) 因此取权重时要转置
            if hf_name in state_dict:
                if "attn.c_attn.weight" in hf_name:

                    weight = state_dict[hf_name].t()
                    
                    split_size = weight.size(0) // 3    
                    q_weight,k_weight,v_weight = torch.split(weight,split_size,dim = 0) 

                    dim = 0
                    chunk_size = q_weight.shape[dim] // tp_size
                    start = tp_rank * chunk_size
                    end = (tp_rank + 1) * chunk_size

                    sharded_q_proj_slice = q_weight[start:end].contiguous()
                    sharded_k_proj_slice = k_weight[start:end].contiguous()
                    sharded_v_proj_slice = v_weight[start:end].contiguous()

                    sharded_c_attn = torch.cat((sharded_q_proj_slice,sharded_k_proj_slice,sharded_v_proj_slice),dim = 0)
                    #print(f"\n\n sharded_c_attn.shape = {sharded_c_attn.shape} \n\n")
                    new_state_dict[my_name] = sharded_c_attn
                    

                elif "attn.c_attn.bias" in hf_name:
                        
                    bias = state_dict[hf_name]
                    split_size = bias.size(0) // 3
                    q_bias,k_bias,v_bias = torch.split(bias,split_size,dim = 0)

                    dim = 0
                    chunk_size = q_bias.shape[dim] // tp_size
                    start = tp_rank * chunk_size
                    end = (tp_rank + 1) * chunk_size

                    sharded_q_proj_slice = q_bias[start:end].contiguous()
                    sharded_k_proj_slice = k_bias[start:end].contiguous()
                    sharded_v_proj_slice = v_bias[start:end].contiguous()

                    sharded_c_attn = torch.cat((sharded_q_proj_slice,sharded_k_proj_slice,sharded_v_proj_slice),dim = 0)

                    new_state_dict[my_name] = sharded_c_attn


                elif "attn.c_proj.weight" in hf_name:
                    #new_state_dict[my_name] = state_dict[hf_name].t()
                    dim = 1
                    full_tensor = state_dict[hf_name].t()
                    chunk_size = full_tensor.shape[dim] // tp_size
                    start = chunk_size * tp_rank
                    end = chunk_size * (tp_rank + 1)
                    sharded_tensor_slice = full_tensor[:,start:end].contiguous()
                    new_state_dict[my_name] = sharded_tensor_slice

                
                elif "mlp.c_fc.weight" in hf_name:
                    #new_state_dict[my_name] = state_dict[hf_name].t()  #no TensorParallel load

                    dim = 0     #先列并行
                    full_tensor = state_dict[hf_name].t()
                    chunk_size = full_tensor.shape[dim] // tp_size
                    start = chunk_size * tp_rank
                    end = chunk_size * (tp_rank + 1)
                    sharded_tensor_slice = full_tensor[start:end,:].contiguous()
                    new_state_dict[my_name] = sharded_tensor_slice

                elif "mlp.c_fc.bias" in hf_name:
                    #列并行时 偏置也需要切分 但行并行的偏置不需要切分
                    
                    dim = 0
                    full_tensor = state_dict[hf_name].t()
                    chunk_size = full_tensor.shape[dim] // tp_size
                    start = chunk_size * tp_rank
                    end = chunk_size * (tp_rank + 1)
                    sharded_tensor_slice = full_tensor[start:end].contiguous()
                    new_state_dict[my_name] = sharded_tensor_slice

                
                elif "mlp.c_proj.weight" in hf_name:
                    #new_state_dict[f"{my_name}"] = state_dict[hf_name].t()     #no TensorParallel load

                    dim = 1
                    full_tensor = state_dict[hf_name].t()
                    chunk_size = full_tensor.shape[dim] // tp_size
                    start = chunk_size * tp_rank
                    end = chunk_size * (tp_rank + 1)
                    sharded_tensor_slice = full_tensor[:,start:end].contiguous()
                    new_state_dict[my_name] = sharded_tensor_slice

                else :
                    new_state_dict[my_name] = state_dict[hf_name] 
            
        #gpt2没有lm_head的权重，因此需要手动加入嵌入层的权重 tie weight 
        if "token_embedding_table.weight" in new_state_dict :
            new_state_dict["lm_head.weight"] = new_state_dict["token_embedding_table.weight"]
    
        model.load_state_dict(new_state_dict,strict=False)
        #self.model.load_state_dict(new_state_dict)
        '''
        strict默认为true,会检测出哪些权重未被加载
        但会检测mask 但mask是非学习的 因此在除了mask外若没有其他未加载参数 即可将strict设为false 
        '''
        model.eval()