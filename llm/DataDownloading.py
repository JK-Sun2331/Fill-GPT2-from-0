from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# 自动下载并加载模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 如果遇到pad_token问题，添加以下代码
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 使用模型进行推理
def generate_text(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 测试
result = generate_text("The future of artificial intelligence")
print(result)