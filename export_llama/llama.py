from transformers import AutoTokenizer
import transformers
import torch

model = "/run/llama-chat-7b-hf"
# model="/root/zanilia/tiny-llama"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
# tokenizer.encode()
from llmint8.quantize import replace_linear_modules
replace_linear_modules(pipeline.model.model)
sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=256,
    temperature=0.7,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

