atc --framework=5 --model="./tiny-llama.onnx"  --output="tiny-llama" --input_format=ND --input_shape="input_ids:1,16;attention_mask:1,1040;position_ids:1,16;past_key_values:22,2,1,4,1024,64" --log=debug --soc_version=Ascend310B1

atc --framework=5 --model="xxx.onnx"  --output="xxx" --input_format=ND --input_shape="input_ids:batch,seq_len;attention_mask:batch,seq_len+kv_len;position_ids:batch,seq_len;past_key_values:n_layer,2,batch,n_head,kv_len,head_dim" --log=debug --soc_version=Ascend310B1 --precision_mode=must_keep_origin_dtype
