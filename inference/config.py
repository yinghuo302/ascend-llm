from dataclasses import dataclass,field
from typing import Optional,Union,List

@dataclass
class InferenceConfig:
    tokenizer: str = ""
    sampling_method: str = "top_k"
    sampling_value: float = 10
    temperature: float = 0.7
    max_length:int = 512 # 输出长度的最大值
    max_input_len:int = 1 # 每次推理输入的最大长度为max_input_len，对om目前设置为1
    session_type:str="acl" # onnx或者acl
    device:int=0
    prompt:str=""
    # prompt:str="[|Human|]Hey there I am a human that would like to have a conversation with you.\n[|AI|]Sure, I am happy to answer most questions\n[|Human|]Great, I insist that we take turns.\n[|AI|]I agree, we should take turns."
    model:str=""
    kvcache_method:str = "sliding-window" # "basic"|"sliding-window"|'streamllm'|'H2O'
    kvcache_fixsize:bool = True # 输入的kv缓存是否固定shape
    head_len:int= 32 # 在KVCache evict时前head len会被保留
    recent_len:int = 32 # 在KVCache evict时最近recent len会被保留
    evict_len:int = 64 # KVCache 逐出的最小值，当KVCache达到最大值时将逐出evict_len个KVCache
    n_layer:int = 22
    format:str='huggingface-tensor' #KVcache的格式
    max_cache_size=256 # kvcache的最大长度
    head_num:int=4
    num_kv_group:int = 8 # for GQA
    head_dim:int=64
    hidden_dim:int=2048
    dtype:str="float16"
    
    def __post_init__(self):
        self.evict_len = int(min((self.max_cache_size - self.head_len )/2,self.evict_len ))
        self.max_input_len = int(min(self.max_input_len,self.evict_len))
        if self.kvcache_method == "streamllm":
            assert(self.head_len+self.evict_len < self.max_cache_size)
        if self.kvcache_method == "H2O":
            self.evict_len = int(min((self.max_cache_size - self.head_len -self.recent_len )/2,self.evict_len ))
            assert(self.head_len+self.recent_len+self.evict_len < self.max_cache_size)
