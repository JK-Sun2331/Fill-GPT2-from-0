import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Dataset
import torch.utils.data as DataLoader
from dataclasses import dataclass
import tiktoken

import os
from safetensors.torch import load_file


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

class Attention(nn.Module):

    def __init__(self,config):
        super().__init__()

        self.head_dim = config.head_dim
        self.all_head_size = config.all_head_size
        self.hidden_size = config.hidden_size
        self.max_seq_len = config.max_seq_len
        self.n_head = config.n_head

        self.key = nn.Linear(self.hidden_size,self.all_head_size)
        self.value = nn.Linear(self.hidden_size,self.all_head_size)
        self.query = nn.Linear(self.hidden_size,self.all_head_size)

        self.proj = nn.Linear(self.hidden_size,self.hidden_size)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        '''
        self.register_buffer(...) 是 PyTorch 的一个巧妙设计。
        它用来注册一个不需要计算梯度、不被视为模型参数（不会在反向传播中更新），但又希望成为模型状态一部分的张量（比如，保存模型时它会被一同保存）。
        为什么用它？ 因为这个掩码矩阵是固定的，不需要学习，所以不作为 parameter 可以节省内存/显存，并可能提高速度。
        '''
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.max_seq_len,self.max_seq_len,device=config.device).view(1,1,self.max_seq_len,self.max_seq_len)) #下三角
        )

    def transpose_for_scores(self,x):
        new_x_shape = x.size()[:-1] + (self.n_head,self.head_dim) #(batch,seq_len,768) -> (batch,seq_len,12,64)
        x = x.view(new_x_shape)
        return x.permute(0,2,1,3) #维度置换 (batch,seq_len,n_head,head_dim) -> (batch,n_head,seq_len,head_dim)

    def forward(self,hidden_state):
        batch_size,seq_len,_ = hidden_state.size()

        query_layer = self.query(hidden_state)
        key_layer = self.key(hidden_state)
        value_layer = self.value(hidden_state)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        attention_scores = torch.matmul(query_layer,key_layer.transpose(-1,-2)) #将key的后两维转置
        #torch.matmul()处理高维矩阵乘法：当输入维度高于2时会执行批量矩阵乘法 (...,m,p) * (...,p,n) = (...,m,n)
        #attention_scores = (batch,n_head,seq_len,head_dim) * (batch,n_head,head_dim,seq_len) = (batch,n_head,seq_len,seq_len)
        attention_scores = attention_scores / (self.head_dim ** 0.5)

        mask = self.mask[:,:,:seq_len,:seq_len]
        attention_scores = attention_scores.masked_fill(mask == 0,torch.finfo(attention_scores.dtype).min)
        #print(attention_scores)
        '''
        masked_fill 方法：
        当 mask == 0 为 True 的位置，用最小值填充
        当 mask == 0 为 False 的位置，保持原值
        '''

        attention_scores = F.softmax(attention_scores,dim=-1)
        attention_scores = self.attn_dropout(attention_scores)

        attention_scores = torch.matmul(attention_scores,value_layer)   #(batch,n_head,seq_len,head_dim)
        attention_scores = attention_scores.permute(0,2,1,3).contiguous()   #(batch,seq_len,n_head,head_dim)
        new_attention_scores_shape = attention_scores.size()[:-2] + (self.all_head_size,)   
        attention_scores = attention_scores.view(new_attention_scores_shape)    #(batch,seq_len,hidden_dim)

        output = self.proj(attention_scores)    #(bacth,seq_len,hidden_dim)
        output = self.resid_dropout(output)
        return output

    
class MLP(nn.Module):

    def __init__(self,config):
        super().__init__()

        self.net1 = nn.Linear(config.hidden_size,config.hidden_size * 4)
        self.activation = nn.GELU()
        self.net2 = nn.Linear(config.hidden_size * 4,config.hidden_size)
        self.dropout = nn.Dropout(config.resid_pdrop)
    
    def forward(self,x):
        
        output = self.net1(x)
        output = self.activation(output)
        output = self.net2(output)
        output = self.dropout(output)
        return output
    

class Block(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.att = Attention(config)
        self.ffn = MLP(config)
        self.ln1 = nn.LayerNorm(config.hidden_size,eps=1e-05)
        self.ln2 = nn.LayerNorm(config.hidden_size,eps=1e-05)

    def forward(self,x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
class GPT2(nn.Module):
    '''
    GPT2和当前LLM的区别主要在于:
        nn.Embedding -> RoPE
        LayerNorm -> RMSNorm
        tie weight
        其他地方差别不大
    '''

    def __init__(self,config):
        super().__init__()
        self.max_seq_len = config.max_seq_len

        self.token_embedding_table = nn.Embedding(config.vocab_size,config.n_embd)
        self.position_embedding_table = nn.Embedding(config.max_seq_len,config.n_embd)
        self.blocks = nn.Sequential(
           * [Block(config) for _ in range(config.n_layer)]
        )
        self.ln_final = nn.LayerNorm(config.n_embd,eps=1e-05)
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias = False)

        self.token_embedding_table.weight = self.lm_head.weight     #tie weight gpt2源码没有

    
    def forward(self,input_ids):

        batch,seq_len = input_ids.size()      #(batch,seq_len)
        #print(f"seq_len = {seq_len},input_ids.size() = {input_ids.size()}")
        assert seq_len <= self.max_seq_len , f"序列长度超出最大序列长度 seq_len = {seq_len},max_seq_len = {self.max_seq_len}"

        token_embd = self.token_embedding_table(input_ids)    #(batch,seq_len,n_embd)
        pos_embd = self.position_embedding_table(
            torch.arange(seq_len,device=input_ids.device)
        )
        x = token_embd + pos_embd       #(batch,seq_len,n_embd)
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)        #(batch,seq_len,vocab_size)
        return logits
    
    
class LLMEngine:

    def __init__(self,model_path : str,device : int):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = GPTConfig
        self.model = GPT2(self.config).to(self.device)
        self.tokenizer = tiktoken.get_encoding("gpt2")
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
            weight_mapping[f"blocks.{i}.att.query.weight"] = f"h.{i}.attn.c_attn.weight"
            weight_mapping[f"blocks.{i}.att.query.bias"] = f"h.{i}.attn.c_attn.bias"
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
                    #在gpt2中 QKV权重存在一个矩阵里，因此要切分
                    weight = state_dict[hf_name]
                    split_size = weight.size(1) // 3    #(768,2304)
                    q_weight,k_weight,v_weght = torch.split(weight,split_size,dim = 1)

                    new_state_dict[f"{my_name}"] = q_weight.t()
                    new_state_dict[my_name.replace("query","key")] = k_weight.t()
                    new_state_dict[my_name.replace("query","value")] = v_weght.t()
                
                elif "attn.c_attn.bias" in hf_name:
                    bias = state_dict[hf_name]
                    split_size = bias.size(0) // 3
                    q_bias,k_bias,v_bias = torch.split(bias,split_size,dim = 0)
                    new_state_dict[f"{my_name}"] = q_bias
                    new_state_dict[my_name.replace("query","key")] = k_bias
                    new_state_dict[my_name.replace("query","value")] = v_bias
                
                
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

    def generator(self,text : str,temperature : float,top_k : int):
        
        print(f"input_text:{text}")

        input_ids = self.tokenizer.encode(text)
        input_len = len(text)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        for i in range(512):
            
            with torch.no_grad():
                logits = self.model(input_ids)      #(batch,seq_len,vocab_size)
                next_token_logits = logits[:,-1,:] / temperature

                if top_k > 0:
                    values, indices = torch.topk(next_token_logits, top_k)  #value返回降序大小为topk的数组,indice是其在原数据的标号
                    #print(f"i_value : {values}")   #基本全是负数 输出可能不正常
                    next_token_logits[next_token_logits < values[:, -1, None]] = -float("inf")

                next_token_logits = F.softmax(next_token_logits,dim = -1)
                #next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) #贪婪解码 效果非常差

                next_token = torch.multinomial(next_token_logits, num_samples=1).to(self.device)
                #print(f"next token = {next_token.item()}")
                if next_token >= self.config.vocab_size:
                    # 如果无效，选择概率最高的有效token
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                if next_token.item() == self.eos_token_id:
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    print("get eos token id")
                    break

                input_ids = torch.cat([input_ids, next_token], dim=1)
        generated_ids = input_ids[0].tolist()
        output_text = self.tokenizer.decode(generated_ids)
        return output_text[input_len:]