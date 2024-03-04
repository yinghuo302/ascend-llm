from typing import List
import numpy as np
from config import InferenceConfig
from inference import LlamaInterface
# cfg=InferenceConfig(tokenizer="/root/zanilia/llama_onnx/tokenizer.model",model="/run/llama-onnx/llama.onnx")
cfg=InferenceConfig(tokenizer="/root/zanilia/tiny-llama",model="/run/llama-onnx/tiny-llama.onnx")
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