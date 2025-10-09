from transformers import GPT2Model, GPT2Config

# 如果是从本地目录加载，指定目录路径
model_path = "/data1/hfhub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

# 加载配置和模型
config = GPT2Config.from_pretrained(model_path)
model = GPT2Model.from_pretrained(model_path)

# 打印模型结构
print(model)