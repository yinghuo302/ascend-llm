# inference

onnx和om模型推理代码，有两种模型
1. cli模式：`python main.py cli`，在终端运行，每一次输入一行，一次性返回所有的推理结果。
2. web模型：`python main.py`，前端代码在[ascend-llm-web]()，打包出dist文件夹，放在当前文件夹下即可。

代码需要修改的部分主要在与config.py，可以根据注释修改。inference.py中关于输入格式和结束语判断的部分可能也需要根据具体的模型修改