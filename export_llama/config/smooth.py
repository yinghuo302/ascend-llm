def get(model_cfg,act_max):
	quant_cfg = {}
	for i in range(model_cfg.num_hidden_layers):
		for name in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj"]:
			quant_cfg[str(i)+"."+name] = {"type":"W8X8"}
		# 对某一个具体的层加act_scale的作用： 若为W8X8，则对该层进行smooth；如为W8SD，则用act_scale进行混合精度分解。
		quant_cfg[str(i)+".down_proj"] = {"type":"W8X8","act_scale":True} 
	quant_cfg["lm_head"] = {"type":"W8X8"}
	quant_cfg["act_scales_path"] = act_max
	quant_cfg["smooth"] = True # 整体的smooth控制是将激活值的缩放与RMSNorm融合，不会造成额外的开销，但down_proj层无法使用
	return quant_cfg