import torch
import torch.nn as nn
import torch.nn.functional as F


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