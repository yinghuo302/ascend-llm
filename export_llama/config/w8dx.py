# 动态混合精度分解
def get(model_cfg,act_max):
	quant_cfg = {}
	for i in range(model_cfg.num_hidden_layers):
		for name in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]:
			quant_cfg[str(i)+"."+name] = {"type":"W8DX"}
	# quant_cfg["lm_head"] = {"type":"W8DX"}  # 可以根据需要取消注释
	# quant_cfg["act_scales_path"] = act_max # 可以根据需要取消注释
	# quant_cfg["smooth"] = True # 可以根据需要取消注释
	return quant_cfg