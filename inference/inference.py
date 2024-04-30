import numpy as np
import os
from typing import Any, Generator, List,Tuple
from threading import Lock
from session import Session
from config import InferenceConfig

class LlamaInterface:
    def __init__(self,config:InferenceConfig) -> None:
        self.max_length = config.max_length
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(config.tokenizer)
        self.sampling_method=config.sampling_method
        self.sampling_value = config.sampling_value
        self.temperature=config.temperature
        self.session=Session.fromConfig(config)
        self.prompt=config.prompt
        self.state:dict[str,Any] = {"code":200,"isEnd":False,"message":""}
        self.reset()
        self.lock = Lock()
        self.first=True
        self.stop_mp = {"[|Human|]":6,"[|AI|]":5,"<|assistant|>":6,"<|user|>":5}
        print("init success")

    def generate_cache(self,prompt:str):
        if len(prompt) == 0 :
            return
        self.first = False
        input_ids = np.asarray(self.tokenizer.encode(prompt),dtype=np.int64).reshape(1,-1)
        logits = self.session.run(input_ids)[0]
        return self.sample_logits(logits[0][-1:],self.sampling_method,self.sampling_value,self.temperature),logits

    def sample_logits(
        self,
        logits: np.ndarray,
        sampling_method: str = "greedy",
        sampling_value: float = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        if temperature == 0 or sampling_method == "greedy":
            next_token = np.argmax(logits, axis=-1).astype(np.int64)

        elif sampling_method == "top_k" or sampling_method == "top_p":
            assert sampling_value is not None
            logits = logits.astype(np.float32)
            logits /= temperature
            probs = np.exp(logits) / np.sum(np.exp(logits))
            sorted_probs = np.sort(probs)[:, ::-1]
            sorted_indices = np.argsort(probs)[:, ::-1]

            if sampling_method == "top_k":
                index_of_interest = int(sampling_value)
            elif sampling_method == "top_p":
                p = sampling_value
                cumulative_probs = np.cumsum(sorted_probs, axis=-1)
                for index_of_interest, cumulative_prob in enumerate(
                    cumulative_probs[0]
                ):
                    if cumulative_prob > p:
                        break

            probs_of_interest = sorted_probs[:, : index_of_interest + 1]
            indices_of_interest = sorted_indices[:, : index_of_interest + 1]
            probs_of_interest /= np.sum(probs_of_interest)
            next_token = np.array(
                [np.random.choice(indices_of_interest[0], p=probs_of_interest[0])]
            )
        else:
            raise Exception(f"Unknown sampling method {sampling_method}")

        return next_token

    def predict(self, text):
        with self.lock:
            self.state['isEnd'],self.state['message'] = False,""        
        if text == "":
            return
        text = preprocess(text)
        input_ids = self.tokenizer.encode(text)
        if not self.first:
            input_ids = [29871,13,29966] + input_ids[2:] # 暂时按tiny llama的tokenizer写死token id
        self.first = False
        input_ids = np.asarray(input_ids,dtype=np.int64).reshape(1,-1)
        ids_list = []
        for i in range(self.max_length):
            logits = self.session.run(input_ids)[0]
            input_ids = self.sample_logits(logits[0][-1:], self.sampling_method, self.sampling_value, self.temperature)
            input_ids = input_ids.reshape(1, -1)
            with self.lock:
                if input_ids[0] == self.tokenizer.eos_token_id:
                    self.state['message'],self.state['isEnd'] = self.tokenizer.decode(ids_list),True
                    break
                ids_list.append(input_ids[0].item())
                text_out = self.tokenizer.decode(ids_list)
                # stop_word = is_stop_word_or_prefix(text_out, ["[|Human|]", "[|AI|]"])
                stop_word = is_stop_word_or_prefix(text_out,["<|user|>","<|assistant|>"])
                if stop_word != "":
                    self.state['message'],self.state['isEnd'] = text_out[:-len(stop_word)].strip(),True
                    #!将结束符对应的KVCache rollback
                    self.session.rollback(self.stop_mp[stop_word]) 
                    break
                self.state['message']=text_out
        with self.lock:
            self.state['isEnd'] = True 
        return self.state['message']

    def reset(self):
        self.first = True
        self.session.reset()
        self.generate_cache(self.prompt)
        
    def getState(self):
        with self.lock:
            return self.state.copy()

#!将输入转换为指定格式
def preprocess(text:str) -> str:
    # return f"[|Human|]{text}\n[|AI|]" 
    return f'<|user|>\n{text}</s>\n<|assistant|>'

#!判断是否为结束语
def is_stop_word_or_prefix(s: str, stop_words: list) -> int:
    for stop_word in stop_words:
        if s.endswith(stop_word):
            return stop_word
    return ""