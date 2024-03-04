import numpy as np
import os
from sentencepiece import SentencePieceProcessor
from typing import Generator, List,Tuple
import gc
from transformers import LlamaTokenizer
from enum import Enum

class State(Enum):
    Generating=0
    EmptyText = 1
    NoModel=2
    LongInput=3
    Success=4
    Stop=5

class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

from session import Session
from config import InferenceConfig
class LlamaInterface:
    def __init__(self,config:InferenceConfig) -> None:
        self.max_length = config.max_length
        # self.tokenizer=Tokenizer(config.tokenizer)
        self.tokenizer:LlamaTokenizer=LlamaTokenizer.from_pretrained(config.tokenizer)
        self.sampling_method=config.sampling_method
        self.sampling_value = config.sampling_value
        self.temperature=config.temperature
        self.session=Session.fromConfig(config)
        self.prompt=config.prompt
        # self.embeddingLayer = torch.nn.Embedding(
        #     self.tokenizer.n_words, self.hidden_size
        # )
        # d = torch.load(config.embedding_file)
        # self.embeddingLayer.load_state_dict(d)
        # self.embeddingLayer.eval()
        self.clean_history()
        print("init success")


    def generate_cache(self,prompt:str):
        if len(prompt) == 0 :
            return
        input_ids = np.asarray(self.tokenizer.encode(prompt, bos=True, eos=False),dtype=np.int64).reshape(1,-1)
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
        if text == "":
            yield "", State.EmptyText
            return
        text = preprocess(text)
        input_ids, _ = self.generate_cache(text)
        input_ids = input_ids.reshape(1, -1)
        for i in range(self.max_length):
            logits = self.session.run(input_ids)[0]
            input_ids = self.sample_logits(logits[0][-1:], self.sampling_method, self.sampling_value, self.temperature)
            input_ids = input_ids.reshape(1, -1)

            # Stop if/when we get an ENDOFTEXT token before reaching maximum sequence length
            if input_ids[0] == self.tokenizer.eos_token_id:
                yield "",State.Success
                del logits
                gc.collect()
                return
            
            text_out = self.tokenizer.decode(input_ids[0].item())
            if is_stop_word_or_prefix(text_out, ["[|Human|]", "[|AI|]"]) is False:
                if "[|Human|]" in text_out:
                    text_out = text_out[: text_out.index("[|Human|]")].strip()
                if "[|AI|]" in text_out:
                    text_out = text_out[: text_out.index("[|AI|]")].strip()
                text_out = text_out.strip()
                yield text_out, State.Generating
            else:
                yield text_out,State.Success
                return
            
        yield "",State.Success

    def clean_history(self):
        self.session.reset()
        self.generate_cache(self.prompt)


def preprocess(text:str) -> str:
    return text

def is_stop_word_or_prefix(s: str, stop_words: list) -> bool:
    for stop_word in stop_words:
        if s.endswith(stop_word):
            return True
        for i in range(1, len(stop_word)):
            if s.endswith(stop_word[:i]):
                return True
    return False