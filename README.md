## 2025.10.9 
初步实现了gpt2-137M的朴素实现，但回答问题时答非所问

## 2025.10.10
将top-k调整到40-50效果达到正常
模型att的输出与官方attn输出的相对容忍度在1e-05的条件下始终为false，但是可以通过1e-04

## 2025 10.10
初步添加了批处理功能，功能不成熟，输出时暂时不能判断eos

## 2025.10.11
完善了批处理功能

## 2025.10.17
在原模型的基础上加入了KV-cache功能。
对于同样的prompt,decode 900次 由11.56s -> 5.68s 
另外GPT2-137M 存放KV-cache的空间如下计算：
token_kvcache_size = 2 * n_layer * head_dim * 4(f32) = 2 * 12 * 768 * 4 = 73728B
seq_kvcache_size = max_seq_size * token_kvcache_size = 1024 * 73728B = 73728KB
batched_kvcache_size = batch_size * seq_kvcache_size = 12 * 73728KB / 1024 = 864MB