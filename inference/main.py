from typing import List
import numpy as np
from config import InferenceConfig
from inference import LlamaInterface
# cfg=InferenceConfig(tokenizer="/root/model/llama-7b-chat-hf",model="/root/model/llama-seq-1-key-256-int8.om")
cfg=InferenceConfig(tokenizer="/root/model/tiny-llama-1.1B",model="/root/model/tiny-llama-seq-1-key-256-int8.om")
# cfg=InferenceConfig(tokenizer="/root/model/tiny-llama-1.1B",model="/root/model/tiny-llama-onnx-int8/llama.onnx")
infer_engine=LlamaInterface(cfg)


def inference(data:List[np.ndarray],idx:int):
    # data = processor.framesPreprocess(frames)
    # del frames
    # model.inference_async(data,idx)
    pass

def inference_cli():
    while True:
        line = input()
        s=[i[0] for i in infer_engine.predict(line)]
        print("".join(s))
        
        

if __name__ == '__main__':
    inference_cli()