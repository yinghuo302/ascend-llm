from config import InferenceConfig
from kvcache import KVCache
import numpy as np
from typing import List
import time
import sys
class Session:
	def __init__(self,config:InferenceConfig) -> None:
		self.kvCache = KVCache.create(config)
		self.max_len = config.max_input_len

	def run(self,input_ids:np.ndarray):
		pass
	
	@staticmethod
	def fromConfig(config:InferenceConfig) -> 'Session':
		if config.session_type == "onnx":
			return OnnxSession(config)
		elif config.session_type=='acl':
			return AclSession(config)
		else:
			return None
	
	def reset(self):
		self.kvCache.reset()

	def rollback(self,seq_len):
		self.kvCache.rollback(seq_len)

	def evict(self,space_need):
		self.kvCache.evict(space_need)
	
class OnnxSession(Session):
	def __init__(self,config:InferenceConfig)->None:
		super().__init__(config)
		import onnxruntime
		options = onnxruntime.SessionOptions()
		self.llm_session = onnxruntime.InferenceSession(
            config.model,
            sess_options=options,
            providers=[
                "DmlExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

	def run(self,input_ids:np.ndarray):
		seq_len=input_ids.shape[-1]
		l,r,result = 0,self.max_len,None
		while l < seq_len:
			r = min(seq_len,r)
			cache,mask,pos_ids = self.kvCache.getInputs(r-l)
			result = self.llm_session.run(None,{
				"input_ids": input_ids,
				"attention_mask":mask,
				"past_key_values": cache,
				"position_ids": pos_ids,
			})
			# result:  [logits,key_values,attn_scores]
			self.kvCache.update(r-l,result[1],result[2])
			l , r = l+self.max_len , r + self.max_len
		return result

class AclSession(Session):
	context = None
	def __init__(self,config:InferenceConfig)->None:
		super().__init__(config)
		from engine import ACLModel,initResource
		self.context = initResource(config.device)
		self.model = ACLModel(config.model,self.context,mode=config.acl_mode)
		self.input_ids = np.zeros((1,self.max_len),dtype=np.int64)
		if config.acl_mode == 'rc':
			self.input_ids,_,_,self.kvCache.kvCache = self.model.getInputs()

	def run(self,input_ids:np.ndarray):
		seq_len=input_ids.shape[-1]
		l,r,result = 0,self.max_len,None
		while l < seq_len:
			r = min(seq_len,r)
			self.input_ids[:,:r-l] = input_ids[:,l:r]
			cache,mask,pos_ids = self.kvCache.getInputs(self.max_len)
			result:List[np.ndarray] = self.model.inference([self.input_ids,mask,pos_ids,cache])
			# result:  [logits,key_values,attn_scores]
			self.kvCache.update(r-l,result[1],result[2])
			l , r = l+self.max_len , r + self.max_len
		return result