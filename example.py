import llm.LLM as LLM

model_path = "/data1/hfhub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e" 

if __name__ == "__main__":
    input_text = [
        "long long ago",
        "I am watching TV"
        ]

    engine = LLM.LLMEngine(model_path,device=1)
    output = engine.generator(input_text,temperature=0.7,top_k=50)
    
    batch_size = len(input_text)
    
    for i in range(batch_size):
        print(f"\n\noutput_text {i} : {output[i]}\n\n")