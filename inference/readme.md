# inference

目前提供两种运行模式：
1. cli模式：在终端运行，每一次输入一行，一次性返回所有的推理结果。
2. web模式：前端代码在[github](https://github.com/yinghuo302/ascend-llm-web)或者[gitee](https://gitee.com/yinghuo302/ascend-llm-web)，打包出dist文件夹，放在inference文件夹下即可。

```bash
cd inference
python main.py \
	--model <path_to_onnx_or_om_model> \
	--hf-dir <path_to_huggingface_model_dir> \ # 需要tokenizer和模型配置文件，权重不需要
	--engine <acl/onnx> 
	--sampling <greedy/top_p/top_k> --sampling_value <>  --temperature <> # 采样相关配置
	--cli # 添加--cli表示在终端运行
```
代码需要修改的部分主要在与config.py，可以根据注释修改。inference.py中关于输入格式和结束语判断的部分可能也需要根据具体的模型修改。