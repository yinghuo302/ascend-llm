import argparse
import importlib
import torch
import os
from transformers import LlamaForCausalLM, LlamaTokenizer


def export_onnx(base_model,out_path,quant_cfg_path,act_path):
    tokenizer= LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model_cfg=model.model.config
    spec = importlib.util.spec_from_file_location("quant_cfg_module", quant_cfg_path)
    quant_cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quant_cfg_module)
    quantize_cfg = quant_cfg_module.get(model_cfg,act_path)
    from quantize import quantize
    quantize(model,quantize_cfg)
    
    input_names = ["input_ids", "attention_mask", "position_ids","past_key_values"]
    output_names = ["logits","out_key_values","attn_scores"]
    dynamic_axes = {
        "input_ids": { 0: "batch_size", 1: "seq_length" },
        "attention_mask": { 0: "batch_size",1:"all_len" },
        "position_ids": { 0: "batch_size", 1: "seq_length" },
        "past_key_values": { 2: "batch_size", 4: "kv_len" },
    }
    
    batch_size,seq_len,kv_len=1,16,1024
    all_len = seq_len + kv_len
    n_layers,n_heads,hidden_size=model_cfg.num_hidden_layers,model_cfg.num_key_value_heads,model_cfg.hidden_size
    head_dim = int(model_cfg.hidden_size / model_cfg.num_attention_heads)


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

    model.eval()
    torch.onnx.export(
        model,
        f=out_path,
        args=input_args,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
        export_params=True,
    )

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m",
        type=str, 
        default="./model/TinyLlama-1.1B-Chat-v1.0", 
        help="transformers model"
    )
    parser.add_argument(
        "--output","-o",
        type=str,
        default="./model/export_out/tiny-llama.onnx",
        help="where to save onnx model",
    )
    parser.add_argument(
        "--act-path","-a",
        type=str,
        default="./act_scales/llama-2-7b.pt",
        help="path to act_scales",
    )
    parser.add_argument(
        "--quant","-q",
        type=str,
        default="./config/w8x8.py",
        help="path to quant config",
    )
    args = parser.parse_args()
    export_onnx(args.model,args.output,args.quant,args.act_path)
