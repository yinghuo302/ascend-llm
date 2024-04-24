quantize_cfg = {
    "q_proj":{
        "type":"W8X8",
    },"k_proj":{
        "type":"W8X8",
    },"v_proj":{
        "type":"W8X8",
    },"o_proj":{
        "type":"W8X8",
    },"gate_proj":{
        "type":"W8X8",
    },"up_proj":{
        "type":"W8X8",
    },
    # "down_proj":{
    #     "type":"W8X8",
    #     "act_scale":True,
    #     "alpha":0.85
	# }
	# "down_proj":{
    #     "type":"W8X8",
    # },"lm_head":{
    #     "type":"W8X8",
    # },
	# "smooth":True,
}