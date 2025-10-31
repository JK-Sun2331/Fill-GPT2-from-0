import llm.LLM as LLM
import time

model_path = "/data1/hfhub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e" 

if __name__ == "__main__":
    input_text = [
        "I have 10 dollar",
        "I am watching TV"
        ]

    batch_size = len(input_text)
    
    engine = LLM.LLMEngine(model_path,batch_size)
    start = time.time()
    output = engine.generator(input_text,temperature=0.7,top_k=50)
    end = time.time()
    
    for i in range(batch_size):
        print(f"\n\noutput_text {i} : {output[i]}\n\n")

    print(end - start)  

    # CUDA_VISIBLE_DEVICES="3,4" torchrun --nproc_per_node=2 example.py