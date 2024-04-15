import numpy as np
from typing import Optional,Tuple,List
from config import InferenceConfig
# 对KV缓存和输出输出格式进行管理
class KVCache:
    def __init__(self,cfg:InferenceConfig) -> None:
        self.head_len = cfg.head_len
        self.max_size = cfg.max_cache_size
        self.input_pos =  0
        self.kv_size = 0
        self.n_layer = cfg.n_layer
        self.kvCache = None
        self.format=cfg.format
        self.head_num=cfg.head_num
        self.head_dim=cfg.head_dim
        self.real_kv_size=0
        self.dtype=np.float16
        if cfg.dtype == "float16":
            self.dtype=np.float16
        elif cfg.dtype=="float32":
            self.dtype=np.float32

    def createEmptyCache(self):
        if self.format == "huggingface-tensor":
            self.kvCache=np.zeros((self.n_layer,2,1,self.head_num,self.max_size,self.head_dim),dtype=self.dtype)
        elif self.format == "huggingface-list":
            self.kvCache=[]
            for i in range(self.n_layer):
                self.kvCache.append([np.zeros((1,self.head_num,self.max_size,self.head_dim),dtype=self.dtype),np.zeros((1,self.head_num,self.max_size,self.head_dim),dtype=self.dtype)])
        elif self.format == 'seq_nhead_headdim':
            self.kvCache = [np.zeros((1,self.n_layer,self.max_size,self.head_num,self.head_dim),dtype=self.dtype),np.zeros((1,self.n_layer,self.max_size,self.head_num,self.head_dim),dtype=self.dtype)]
        elif self.format == 'nhead_seq_headdim':
            self.kvCache = [np.zeros((1,self.n_layer,self.head_num,self.max_size,self.head_dim),dtype=self.dtype),np.zeros((1,self.n_layer,self.head_num,self.max_size,self.head_dim),dtype=self.dtype)]

    def update(self,seq_len:int,newKV:Tuple[List[np.ndarray],List[np.ndarray]],scores:Optional[np.ndarray]=None)->None:
        pass
    
    def evict(self,space_need):
        pass

    def getInputs(self, seq_len: int) -> List[np.ndarray]:
        cache = self.kvCache[:,:,:,:,0:self.kv_size]
        mask = np.ones((1,self.kv_size+seq_len),dtype=np.int64)
        mask[:,self.real_kv_size:self.kv_size] = 0
        pos_id =np.arange(self.input_pos,self.input_pos+seq_len,dtype=np.int64).reshape(1,-1)
        return cache,mask,pos_id
    
    def reset(self):
        self.input_pos=0
        self.real_kv_size=0
        self.createEmptyCache()
    
    @staticmethod
    def create(config:InferenceConfig) -> 'KVCache':
        if config.kvcache_method == "basic":
            return Basic(config)
        elif config.kvcache_method == "fixsize":
            return FixSize(config)
        elif config.kvcache_method == 'streamllm':
            return FixSizeStreamLLM(config)
        elif config.kvcache_method == 'H2O':
            return FixSizeH2O(config)
        else:
            return None

class Basic(KVCache):
    def __init__(self, cfg: InferenceConfig) -> None:
        super().__init__(cfg)
        
    def update(self, seq_len: int, newKV: Tuple[List[np.ndarray]], scores: Optional[np.ndarray] = None) -> None:
        if seq_len + self.kv_size > self.max_size:
            raise RuntimeError("超出KV缓存长度限制")
        if self.format=="huggingface-tensor":
            newKV.reshape(self.n_layer,2,1,self.head_num,-1,self.head_dim)
            self.kvCache[:,:,:,:,self.kv_size:self.kv_size+seq_len,:] = newKV[:,:,:,:,0:seq_len,:]
        self.kv_size += seq_len
        self.input_pos+=seq_len
        self.real_kv_size += seq_len
    
    def reset(self):
        self.kv_size=0
        return super().reset()

class FixSize(KVCache):
    def __init__(self, cfg: InferenceConfig) -> None:
        super().__init__(cfg)
        self.kv_size=self.max_size
        
    def update(self, seq_len: int, newKV: Tuple[List[np.ndarray]], scores: Optional[np.ndarray] = None) -> None:
        self.input_pos += seq_len
        if seq_len + self.real_kv_size > self.max_size:
            seq_len = self.max_size - self.real_kv_size
        if seq_len <= 0:
            return
        if self.format=="huggingface-tensor":
            newKV = newKV.reshape(self.n_layer,2,1,self.head_num,-1,self.head_dim)
            self.kvCache[:,:,:,:,self.real_kv_size:self.real_kv_size+seq_len,:] = newKV[:,:,:,:,0:seq_len,:]
        self.real_kv_size += seq_len

class FixSizeStreamLLM(KVCache):
    def __init__(self,cfg:InferenceConfig) -> None:
        super().__init__(cfg)
        self.p=0
        self.kv_size=self.max_size

    def update(self,seq_len:int,newKV:Tuple[List[np.ndarray],List[np.ndarray]],score:Optional[np.ndarray] = None):
        self.input_pos+=seq_len
        cur = 0
        while self.p + seq_len  > self.max_size:
            self.update_part(newKV,cur,self.max_size-self.p)
            cur += (self.max_size-self.p)
            seq_len -= (self.max_size-self.p)
            self.p = self.head_len
            self.real_kv_size = self.max_size
        self.update_part(newKV,cur,seq_len)
        self.p += seq_len
        self.real_kv_size = max(self.p,self.real_kv_size)

    def update_part(self,newKV:Tuple[List[np.ndarray],List[np.ndarray]],begin:int,len:int):
        if len == 0:
            return
        if self.format == 'huggingface-tensor': #[n_layer,2,batch_size,head_num,len,head_dim]
            self.kvCache[:,:,:,:,self.p:self.p+len,:] = newKV[:,:,:,:,begin:begin+len,:]	
        if self.format=='seq_nhead_headdim': # [batch, n_layers, seq_len, n_heads, head_dim]
            self.kvCache[0][:,:,self.p:self.p+len] = newKV[0][:,:,begin:begin+len]
            self.kvCache[1][:,:,self.p:self.p+len] = newKV[1][:,:,begin:begin+len]
        elif self.format=='nhead_seq_headdim':    # [batch, n_layers, n_heads, seq_len, head_dim]
            self.kvCache[0][:,:,:,self.p:self.p+len] = newKV[0][:,:,:,begin:begin+len]
            self.kvCache[1][:,:,:,self.p:self.p+len] = newKV[1][:,:,:,begin:begin+len]
        elif self.format=='huggingface-list': # (n_layer,2) * [batch_size,head_num,len,head_dim]
            for i in range(self.n_layer):
                self.kvCache[i][0][:,:,self.p:self.p+len,:] = newKV[i][0][:,:,begin:begin+len,:]	
                self.kvCache[i][1][:,:,self.p:self.p+len,:] = newKV[i][1][:,:,begin:begin+len,:]
    
    def reset(self):
        self.p=0
        self.real_kv_size = 0
        return super().reset()

# 未完成
# TODO：
class FixSizeH2O(KVCache):
    def __init__(self,cfg:InferenceConfig) -> None:
        super().__init__(cfg)
        self.scores = np.zeros((self.n_layer,1,self.head_num,self.kv_size),dtype=self.dtype)
        self.idx_head = np.arange(0,self.head_num)
    
    def update(self,newKV:Tuple[List[np.ndarray],List[np.ndarray]],score:Optional[np.ndarray] = None):
        seq_len = newKV[0][0].shape[-2]
        for i in range(seq_len):
            self.update_one(newKV[:,:,:,:,i,:],score[:,:,:,i,:])
        
                
    def update_one(self,newKV:Tuple[List[np.ndarray],List[np.ndarray]],score:Optional[np.ndarray] = None):
        if self.real_kv_size <  self.kv_size:
            self.kvCache[:,:,:,:,self.real_kv_size,:] = newKV
            self.real_kv_size += 1
            self.scores[:,:,:,:self.real_kv_size] = self.scores[:,:,:,:self.real_kv_size] * 0.5 + score[:,:,:,:self.real_kv_size]
            return
        score = score.squeeze(-2)
        self.scores[:,:,:,:self.real_kv_size] = self.scores[:,:,:,:self.real_kv_size] * 0.5 + score[:,:,:,:self.real_kv_size]
        for i in range(self.n_layer):
            min_idx = np.argmin(self.scores[i,0],axis=-1)
            self.kvCache[i,0,0,self.idx_head,min_idx] = newKV[i,0,0,self.idx_head,-1]
            self.kvCache[i,1,0,self.idx_head,min_idx] = newKV[i,1,0,self.idx_head,-1]
            self.scores[i,0,self.idx_head,min_idx]=score[i,0,self.idx_head,-1]
            
    