import llm.LLM as LLM
import torch
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel

model_path = "/data1/hfhub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e" 

if __name__ == "__main__":
    input_text = "Introduce US"
    engine = LLM.LLMEngine(model_path,device=1)

    hf_model = GPT2LMHeadModel.from_pretrained("gpt2").to(1)
    hf_model.eval()

    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode("Hello", return_tensors="pt").to(1)

    #检查嵌入层数据是否一致
    _,seq_len = input_ids.size()

    my_output_embd = engine.model.token_embedding_table(input_ids) + engine.model.position_embedding_table(torch.arange(input_ids.shape[1],device = 1))
    hf_output_emb = hf_model.transformer.wte(input_ids) + hf_model.transformer.wpe(torch.arange(input_ids.shape[1],device=1))

    #print(torch.allclose(my_output_embd, hf_output_emb))

    #检查block数据是否一致
    #my_block_ln1_output = engine.model.blocks[0].ln1(my_output_embd)
    #hf_block_ln1_output = hf_model.transformer.h[0].ln_1(hf_output_emb)

    my_block_attn_output = engine.model.blocks[0].att(my_output_embd)
    hf_block_attn_output = hf_model.transformer.h[0].attn(hf_output_emb)[0]


    
    #print("********************************************")
    #print(my_block_attn_output)
    #print("********************************************")
    #print(hf_block_attn_output)
    print(torch.allclose(my_block_attn_output,hf_block_attn_output))#attn输出不一致