import os
import sys
from typing import List
import torch
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer



def export_onnx(
    base_model: str = "/root/zanilia/tiny-llama",
    output_dir: str = "/run/llama-onnx/tiny-llama.onnx",
):

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    input_names = ["input_ids", "attention_mask", "position_ids","past_key_values"]
    output_names = ["logits","out_key_values"]
    dynamic_axes = {
        "input_ids": { 0: "batch_size", 1: "seq_length" },
        "attention_mask": { 0: "batch_size",1:"all_len" },
        "position_ids": { 0: "batch_size", 1: "seq_length" },
        "past_key_values": { 2: "batch_size", 4: "kv_len" },
    }
    cfg=model.model.config
    batch_size,seq_len,kv_len=1,16,1024
    all_len = seq_len + kv_len
    n_layers,n_heads,hidden_size=cfg.num_hidden_layers,cfg.num_key_value_heads,cfg.hidden_size
    head_dim = int(cfg.hidden_size / cfg.num_attention_heads)


    input_ids = torch.zeros((batch_size,seq_len)).long().to("cuda") # batch_size, new_sequence_length
    attention_mask = torch.zeros((batch_size,all_len)).long().to("cuda") # batch_size, all_sequence_length
    position_ids = torch.zeros((batch_size,seq_len)).long().to("cuda") # batch_size, new_sequence_length
    # past_keys = torch.rand((batch_size,  n_heads,kv_len, head_dim),dtype=torch.float16).to("cuda")
    # past_values = torch.rand((batch_size,n_heads, kv_len, head_dim),dtype=torch.float16).to("cuda")
    # past_key_values = tuple([(past_keys,past_values)] * n_layers)
    past_key_values = torch.rand((n_layers,2,batch_size,n_heads, kv_len, head_dim),dtype=torch.float16).to("cuda")
    input_args = (
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        None, # inputs_embeds: Optional[torch.FloatTensor] = None,
        None, #labels: Optional[torch.LongTensor] = None,
        True, #use_cache: Optional[bool] = None,
        True # output_attentions: Optional[bool] = None,
    )


    torch.onnx.export(
        model,
        f=output_dir,
        args=input_args,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        export_params=True
    )
    # import onnx
    # onnx_model = onnx.load(
    #     output_dir, load_external_data=True
    # )  # this will probably be too memory heavy for large models
    # onnx.save(
    #     onnx_model,
    #     output_dir,
    #     save_as_external_data=False,
    # )


if __name__ == "__main__":
    export_onnx()