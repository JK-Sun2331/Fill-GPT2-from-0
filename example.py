import llm.LLM as LLM

model_path = "/data1/hfhub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e" 

if __name__ == "__main__":
    input_text = "Who is the first president in America?"
    engine = LLM.LLMEngine(model_path,device=1)
    output = engine.generator(input_text,temperature=0.7,top_k=4)
    print(f"output_text:{output}")