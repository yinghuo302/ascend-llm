# ascend-llm

在昇腾310上部署大模型，目前已经跑通meta-llama/Llama-2-7b-chat-hf和TinyLlama/TinyLlama-1.1B-Chat-v1.0

### export_llama

将llama模型导出为onnx和量化相关代码。

1. 导出onnx：将transformer库中的model_llama替换为export_llama文件下的[model_llama](./export_llama/modeling_llama_4.35.py)，通过`python export_llama.py`导出onnx，
2. 模型量化：在`export_llama.py`中引用量化的配置文件，调用`quantize`函数即可导出量化后的onnx文件，具体见[readme](./export_llama/readme.md)

### custom_op

在昇腾安装目录tools/msopgen/template/custom_operator_sample/DSL/Onnx有自定义算子模板，将`matmul_integer_plugin.cc`放在算子模板工程目录下的framework/onnx_plugin下，然后进行编译，安装自定义算子，将Onnx的MatmulInteger对应到昇腾CANN的BatchMatmulV2算子。

### inference 

onnx和om模型推理代码，有两种模式
1. cli模式：`cd inference/ && python main.py cli`，在终端运行，每一次输入一行，一次性返回所有的推理结果。
2. web模式：`cd inference/ && python main.py`，前端代码在[github](https://github.com/yinghuo302/ascend-llm-web)或者[gitee](https://gitee.com/yinghuo302/ascend-llm-web)，打包出dist文件夹，放在inference文件夹下即可。

代码需要修改的部分主要在与config.py，可以根据注释修改。inference.py中关于输入格式和结束语判断的部分可能也需要根据具体的模型修改