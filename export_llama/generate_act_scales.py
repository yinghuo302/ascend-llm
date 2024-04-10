'''
code from https://github.com/mit-han-lab/smoothquant/
'''
from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm
import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse

def get_act_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m,torch.nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    dataset = load_dataset("json", data_files=dataset_path,split="train")
    dataset = dataset.shuffle(seed=42)

    for i in tqdm(range(num_samples)):
        text = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n".format(
			instruction=dataset["instruction"][i], input=dataset["output"][i]
		)
        input_ids = tokenizer(
            text, return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales



def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="/run/llama-chat-7b-hf", help="model name"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="act_scales/opt-1.3b.pt",
        help="where to save the act scales",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/root/zanilia/alpaca-lora/alpaca_data.json",
        help="location of the calibration dataset, we use the validation set of the Pile dataset",
    )
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_name)

    act_scales = get_act_scales(
        model, tokenizer, args.dataset_path, args.num_samples, args.seq_len
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)


if __name__ == "__main__":
    main()
