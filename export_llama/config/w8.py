# 仅权重int8量化
def get(model_cfg,act_max):
	quant_cfg = {}
	for i in range(model_cfg.num_hidden_layers):
		for name in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]:
			quant_cfg[str(i)+"."+name] = {"type":"W8"}
	quant_cfg["lm_head"] = {"type":"W8"}
	return quant_cfg