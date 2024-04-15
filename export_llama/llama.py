from transformers import AutoTokenizer
import transformers
import torch
from quantize import quantize

model = "/run/llama-chat-7b-hf"
# model="/root/zanilia/tiny-llama"

tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.pad_token_id = tokenizer.eos_token_id
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
from config.sd import quantize_cfg
quantize(pipeline.model,cfg=quantize_cfg)
# prompt_text = '''### 40 WORD SYNOPSIS of Story Okono
# Okono is the newest hero introduced to League of Legends. His ultimate power electrocutes all enemies nearby. After Okono is added to the game, Nate plays as Okono obsessively. He wins many games and reaches Challenger.
# ### 1000 WORD STORY Okono
# '''
prompt_text="[|Human|] I am going to Paris, what should I see? \n[|AI|]"
sequences = pipeline(
    prompt_text,
    do_sample=True,
    top_k=3,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=256,
    temperature=0.7,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

