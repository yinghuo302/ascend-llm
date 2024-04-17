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
        self.dtype=np.float16
        self.fix_size = cfg.kvcache_fixsize
        self.evict_len = cfg.evict_len
        self.recent_len = cfg.recent_len
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
    
    def evict(self,space_need:int):
        pass

    def getInputs(self, seq_len: int) -> List[np.ndarray]:
        cache,mask = None,None
        if self.fix_size:
            cache,mask = self.kvCache, np.ones((1,self.max_size+seq_len),dtype=np.int64)
            mask[:,self.kv_size:self.max_size] = 0
        else:
            cache,mask = self.kvCache[:,:,:,:,:self.kv_size], np.ones((1,self.kv_size+seq_len),dtype=np.int64)
        pos_id =np.arange(self.input_pos,self.input_pos+seq_len,dtype=np.int64).reshape(1,-1)
        return cache,mask,pos_id
    
    def reset(self):
        self.input_pos=0
        self.kv_size=0
        self.createEmptyCache()
    
    def rollback(self,seq_len):
        self.kv_size -=seq_len

    @staticmethod
    def create(config:InferenceConfig) -> 'KVCache':
        if config.kvcache_method == "basic":
            return Basic(config)
        elif config.kvcache_method == "sliding-window":
            return SWindow(config)
        elif config.kvcache_method == 'streamllm':
            return StreamLLM(config)
        elif config.kvcache_method == 'H2O':
            return H2O(config)
        else:
            return None

class Basic(KVCache):
    def __init__(self, cfg: InferenceConfig) -> None:
        super().__init__(cfg)
        
    def update(self, seq_len: int, newKV: Tuple[List[np.ndarray]], scores: Optional[np.ndarray] = None) -> None:
        if seq_len + self.kv_size > self.max_size:
            raise RuntimeError("超出KV缓存长度限制")
        if self.format=="huggingface-tensor":
            self.kvCache[:,:,:,:,self.kv_size:self.kv_size+seq_len,:] = newKV[:,:,:,:,0:seq_len,:]
        self.kv_size += seq_len
        self.input_pos+=seq_len

class SWindow(KVCache):
    def __init__(self,cfg:InferenceConfig) -> None:
        super().__init__(cfg)
        self.p=0
        self.cnt = 0

    def update(self,seq_len:int,newKV:Tuple[List[np.ndarray],List[np.ndarray]],score:Optional[np.ndarray] = None):
        self.input_pos+=seq_len
        cur = 0
        while self.p + seq_len  > self.max_size:
            self.update_part(newKV,cur,self.max_size-self.p)
            cur += (self.max_size-self.p)
            seq_len -= (self.max_size-self.p)
            self.p = self.head_len
            self.kv_size = self.max_size
            self.cnt += 1
        self.update_part(newKV,cur,seq_len)
        self.p += seq_len
        self.kv_size = max(self.p,self.kv_size)

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
        return super().reset()
    
    def rollback(self, seq_len):
        if self.cnt != 0:
            self.p -= seq_len
            self.cnt -= 1
            if self.p < self.head_len:
                self.p = self.max_size - (self.head_len - self.p) + 1
            if self.cnt == 0:
                self.kv_size = self.p
        else:
            self.p -= seq_len
            self.kv_size -= seq_len

class StreamLLM(KVCache):
    def __init__(self,cfg:InferenceConfig):
        super().__init(cfg)
    
    def update(self,seq_len:int,newKV:Tuple[List[np.ndarray],List[np.ndarray]],score:Optional[np.ndarray] = None):
        if self.kv_size + seq_len >= self.max_size:
            self.evict(self.evict_len)
        self.input_pos += seq_len
        self.kvCache[:,:,:,:,self.kv_size:self.kv_size+seq_len] = newKV

    def evict(self, space_need: int):
        self.kvCache[:,:,:,:,self.head_len:self.kv_size-space_need + self.head_len] = \
            self.kvCache[:,:,:,:,self.head_len+space_need:self.kv_size]

class H2O(KVCache):
    def __init__(self,cfg:InferenceConfig) -> None:
        super().__init__(cfg)
        self.scores = np.zeros((self.n_layer,1,self.head_num,self.max_size),dtype=self.dtype)
        self.idx_head = np.arange(0,self.head_num,dtype=np.int32).reshape(-1,1)
    
    def update(self,seq_len:int,newKV:Tuple[List[np.ndarray],List[np.ndarray]],score:Optional[np.ndarray] = None):
        # score [n_layer,batch,nheader,input_len,all_len]
        score[:,:,:,:,self.kv_size:self.kv_size+seq_len] = score[:,:,:,:,-seq_len:]
        if self.kv_size + seq_len >= self.max_size:
            self.o_score = score
            self.evict(self.evict_len)
            self.o_score,score = None,self.o_score
        self.input_pos += seq_len
        self.kvCache[:,:,:,:,self.kv_size:self.kv_size+seq_len] = newKV
        for i in range(seq_len):
            self.update_score_one(score[:,:,:,i])
                
    def update_score_one(self,score:Optional[np.ndarray] = None):
        self.kv_size += 1
        self.scores[:,:,:,:self.kv_size] = self.scores[:,:,:,:self.kv_size] * 0.5 + score[:,:,:,:self.kv_size]
        
    def evict(self, space_need):
        r_len = self.kv_size - space_need - self.head_len -self.recent_len # 对前head len个KV缓存进行保留
        for i in range(self.n_layer):
            idx=np.argpartition(-self.scores[i,0,:,self.head_len:self.kv_size-self.recent_len],r_len,axis=-1)[:,:r_len]
            for j in range(2):
                self.kvCache[i,j,0,:,self.head_len:self.head_len+r_len] = self.kvCache[i,j,0,self.idx_head,idx]
                self.kvCache[i,j,0,:,self.head_len+r_len:self.kv_size-space_need] = \
                    self.kvCache[i,j,0,:,self.kv_size-self.recent_len:self.kv_size]
            self.scores[i,0,:,self.head_len:r_len+self.head_len] = self.scores[i,0,self.idx_head,idx]
            self.scores[i,0,:,self.head_len+r_len:self.kv_size-space_need] = \
                self.scores[i,0,:,self.kv_size-self.recent_len:self.kv_size]
            new_seq = self.o_score.shape[-2]
            for j in range(new_seq):
                self.o_score[i,0,:,j,self.head_len:r_len+self.head_len] = self.o_score[i,0,self.idx_head,j,idx]
                self.o_score[i,0,:,j,self.head_len+r_len:self.kv_size+new_seq-space_need] = \
                    self.o_score[i,0,:,self.kv_size-self.recent_len:self.kv_size+new_seq]
            self.scores[i,0,:,r_len+self.head_len+self.recent_len:] =  0
            self.kv_size = r_len + self.head_len + self.recent_len
