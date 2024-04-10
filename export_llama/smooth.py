'''
code from https://github.com/mit-han-lab/smoothquant/
'''
import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, (LlamaRMSNorm,nn.Linear))
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.shape[0] == fc.in_features == act_scales.numel()
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    if ln.weight.dim() == 2:
        ln.weight.div_(scales.unsqueeze(-1))
    else:
        ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer):
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
            # smooth_ln_fcs_llama_like(module.mlp.up_proj,module.mlp.down_proj,scales[name + ".mlp.down_proj"],0.9)