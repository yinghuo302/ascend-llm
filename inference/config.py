from dataclasses import dataclass,field
from typing import Optional,Union,List

@dataclass
class InferenceConfig:
    tokenizer: str = ""
    sampling_method: str = "top_k"
    sampling_value: float = 10
    temperature: float = 0.7
    max_length:int = 64 # 输出长度的最大值
    session_type:str="acl"
    device:int=0
    prompt:str=""
    # prompt:str="[|Human|]Hey there I am a human that would like to have a conversation with you.\n[|AI|]Sure, I am happy to answer most questions\n[|Human|]Great, I insist that we take turns.\n[|AI|]I agree, we should take turns.\n[|Human|]Great, can we also keep answers short\n[|AI|]Yes, short answers are usually best"
    model:str=""
    embedding_file=""
    kvcache_method:str = "fixsize"
    head_len:int= 64
    n_layer:int = 22
    format:str='huggingface-tensor' #KVcache的格式
    max_cache_size=256 # kvcache的最大长度
    head_num:int=4
    head_dim:int=64
    hidden_dim:int=2048
    dtype:str="float16"
