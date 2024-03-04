from config import InferenceConfig
from kvcache import KVCache
import numpy as np
class Session:
	def __init__(self,config:InferenceConfig) -> None:
		self.kvCache = KVCache.create(config)

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
		cache,mask,pos_ids = self.kvCache.getInputs(seq_len)
		result = self.llm_session.run(None,{
			"input_ids": input_ids,
			"attention_mask":mask,
			"past_key_values": cache,
			"position_ids": pos_ids,
		})
		self.kvCache.update(seq_len,result[1])
		return result

import math
class AclSession(Session):
	context = None
	def __init__(self,config:InferenceConfig)->None:
		super().__init__(config)
		from engine import AclModel,initResource
		if context is None:
			context = initResource(config.device)
		self.model = AclModel(config.model,context)
		self.input_ids = np.zeros((1,16),dtype=np.float16)

	def run(self,input_ids:np.ndarray):
		seq_len,i = input_ids.shape[-1],0
		logits = []
		while i < seq_len:
			end = i + 16 if i+16 < seq_len else seq_len
			cache,mask,pos_ids = self.kvCache.getInputs(end-i)
			self.input_ids[0:end-i] = input_ids[i:end]
			result = self.model.inference([self.input_ids,mask,pos_ids,cache])
			self.kvCache.update(end-i,result[1])
			logits.append(result[0][0:end-i])
		return [np.concatenate(logits)]