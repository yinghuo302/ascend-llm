# 对down_proj混合精度分解，对其他部分平滑激活
def get(model_cfg,act_max):
    quant_cfg = {}
    d_mx = findN(0.1 * model_cfg.intermediate_size)
    scale,step = 4, 4/model_cfg.num_hidden_layers
    for i in range(model_cfg.num_hidden_layers):
        scale = max(0,scale-step)
        d_cur = max(32,d_mx >> int(scale))
        for name in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj"]:
            quant_cfg[str(i)+"."+name] = {"type":"W8X8"}
        quant_cfg[str(i)+".down_proj"] = {"type":"W8SD","act_scale":True,"alpha":d_cur}
    quant_cfg["lm_head"] = {"type":"W8SD","act_scale":True,"alpha":64}
    quant_cfg["act_scales_path"] = act_max
    quant_cfg["smooth"] = True
    return quant_cfg

def findN(N):
    sum = 1; 
    while True:
        if sum * 2 > N:
            return sum	
        sum = sum * 2