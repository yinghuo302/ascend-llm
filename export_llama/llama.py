from transformers import AutoTokenizer
import transformers
import torch

# model = "/run/llama-chat-7b-hf"
model="/root/zanilia/tiny-llama"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
quantize_cfg = {
    "q_proj":{
        "type":"W8X8",
        "act_scale":False
    },"k_proj":{
        "type":"W8X8",
        "act_scale":False
    },"v_proj":{
        "type":"W8X8",
        "act_scale":False
    },"o_proj":{
        "type":"W8X8",
        "act_scale":False
    },"gate_proj":{
        "type":"W8X8",
        "act_scale":False
    },"up_proj":{
        "type":"W8X8",
        "act_scale":False
    },"down_proj":{
        "type":"W8X8",
        "act_scale":False,
        "alpha":0.85
    },
}
from quantize import quantize
# quantize(pipeline.model,act_scales_path="/root/zanilia/quantize/act_scales/llama-2-7b.pt",cfg=quantize_cfg)
quantize(pipeline.model,cfg=quantize_cfg)
# from quantize import replace_linear_modules
# replace_linear_modules(pipeline.model)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=256,
    temperature=0.7,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

