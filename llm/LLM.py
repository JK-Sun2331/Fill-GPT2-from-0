import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Dataset
import torch.utils.data as DataLoader
from dataclasses import dataclass
#import tiktoken
from transformers import AutoTokenizer
from llm.Attention import GPT2
import os
from safetensors.torch import load_file
import torch.distributed as dist

torch.manual_seed(1024)

@dataclass
class GPTConfig:
    max_seq_len : int = 1024
    batch_size : int = 12
    n_layer : int = 12
    n_head : int = 12
    n_embd :int = 768
    hidden_size : int = 768
    head_dim : int = hidden_size // n_head 
    all_head_size :int = hidden_size
    attn_pdrop : float = 0.1 
    resid_pdrop : float = 0.1
    vocab_size : int = 50257 
    eos_token_id : int = 50256

    device = 1
    
def setup_distributed():

    if not dist.is_initialized():
        #CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 your_inference_script.py
        local_rank = int(os.environ.get('LOCAL_RANK'))
        torch.cuda.set_device(local_rank)   #设置当前进程使用的GPU

        dist.init_process_group(backend = 'nccl')   #初始化进程组
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    
    return rank,world_size




class LLMEngine:

    def __init__(self,model_path : str,device : int):
        self.model_path = model_path
        self.device,self.world_size = setup_distributed()
        self.config = GPTConfig
        self.model = GPT2(self.config).to(self.device)
        #self.tokenizer = tiktoken.get_encoding("gpt2")     #使用tiktoken面对批处理编码时 需要左填充 并设置 mask 很麻烦
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._load_weight(self.model_path)

        self.eos_token_id = self.config.eos_token_id
        self.max_seq_len = self.config.max_seq_len



    
    def _load_weight(self,model_path : str):
        weight_file = os.path.join(model_path,"model.safetensors")
        assert weight_file,"There is no weight_file"

        #加载权重
        state_dict = load_file(weight_file,device=self.device)

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
        for i in range(self.config.n_layer):
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
                    new_state_dict[my_name] = state_dict[hf_name].t()
                
                elif "attn.c_proj.weight" in hf_name:
                    new_state_dict[my_name] = state_dict[hf_name].t()
                
                elif "mlp.c_fc.weight" in hf_name or "mlp.c_proj.weight" in hf_name:
                    new_state_dict[f"{my_name}"] = state_dict[hf_name].t()

                else :
                    new_state_dict[my_name] = state_dict[hf_name] 
            
        #gpt2没有lm_head的权重，因此需要手动加入嵌入层的权重 tie weight 
        if "token_embedding_table.weight" in new_state_dict :
            new_state_dict["lm_head.weight"] = new_state_dict["token_embedding_table.weight"]
    
        self.model.load_state_dict(new_state_dict,strict=False)
        #self.model.load_state_dict(new_state_dict)
        '''
        strict默认为true,会检测出哪些权重未被加载
        但会检测mask 但mask是非学习的 因此在除了mask外若没有其他未加载参数 即可将strict设为false 
        '''
        self.model.eval()


    #def allocate_kvcache(self,config):
        #token_kvcache_size = 2 * n_layer * head_dim * 4(f32) = 2 * 12 * 768 * 4 = 73728B
        #seq_kvcache_size = max_seq_size * token_kvcache_size = 1024 * 73728B = 73728KB
        #batched_kvcache_size = batch_size * seq_kvcache_size = 12 * 73728KB = 864MB



    def text_encoding(self,texts : list[str]):
        tokenizer = self.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        inputs = tokenizer(
            texts, 
            padding=True,       # 开启自动填充
            truncation=True,    # 开启自动截断
            return_tensors="pt" # 返回 PyTorch 张量
            )
        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long, device=self.device)
        return input_ids
    
    def sampling(self,logits,temperature : float,top_k : int):

        next_token_logits = logits[:,-1,:] / temperature

        if top_k > 0:
            values, indices = torch.topk(next_token_logits, top_k)  #value返回降序大小为topk的数组,indice是其在原数据的标号
            next_token_logits[next_token_logits < values[:, -1, None]] = -float("inf")
        next_token_logits = F.softmax(next_token_logits,dim = -1)

        #next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) #贪婪解码 效果非常差
        next_tokens = torch.multinomial(next_token_logits, num_samples=1).to(self.device)

        return next_tokens
    

    @torch.no_grad()
    def generator(self,texts : list[str],temperature : float,top_k : int):
        
        batch_size = len(texts)
        output = []
        output_texts = []

        prompt_ids = self.text_encoding(texts)

        #第一轮先处理prefill
        logits,past_kv = self.model(input_ids = prompt_ids,past_key_value = None)  #(batch,seq_len,vocab_size)
        next_token = self.sampling(logits,temperature,top_k)
        output_ids = next_token
            
        for i in range(900): #输入长度 + 循环次数 不能大于1024 否则会越界
            #循环decode
            logits,past_kv = self.model(input_ids = next_token,past_key_value = past_kv)   #(batch,seq_len,vocab_size)
            next_token = self.sampling(logits,temperature,top_k)
            output_ids = torch.cat([output_ids, next_token],dim = 1)

            mask = (next_token != self.eos_token_id).squeeze(-1)     #如果新生成的token所在行包含eos,那么将其掩盖

            if len(output_ids[~mask].tolist()) != 0:
                output.append(output_ids[~mask].squeeze().tolist())     #将结束的seq输出到output
                
            next_token = next_token[mask]             #将结束的seq移除
            output_ids = output_ids[mask]
    
        
            for ids in mask.flatten():
                if ids.item() == 0:
                    past_kv = list(past_kv)
                    for i,layer in enumerate(past_kv):
                        past_kv[i] = list(past_kv[i])
                        k_cache = layer[0][mask]
                        v_cache = layer[1][mask]
                        past_kv[i] = [k_cache,v_cache]
                        past_kv[i] = tuple(past_kv[i])
                    past_kv = tuple(past_kv)
                    break
                     
        #若循环结束都没生成eos,那么将所有seq输出
        final_len = output_ids.shape[0]

        for i in range(final_len):
            output.append(output_ids[i].tolist())

        for i in range(batch_size):
            generated_ids = output[i]
            output_texts.append(self.tokenizer.decode(generated_ids))
        return output_texts