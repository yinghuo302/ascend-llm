# ascend-llm

## 简介

本项目基于昇腾310芯片部署大语言模型，目前已经成功运行meta-llama/Llama-2-7b-chat-hf和TinyLlama/TinyLlama-1.1B-Chat-v1.0。

本实践项目由南京大学计算机科学与技术系杜骋同学主导，由朱光辉老师进行指导，由昇腾CANN生态使能团队提供技术支持。

## 运行方式

### 环境准备

1. 昇腾软硬件解决方案(驱动+固件+CANN)
   
   前往[昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha002/cannquickstart/quickstart/instg_000021.html)，按照说明下载安装。
   
2. 第三方依赖
   
   模型导出和推理相关文件夹下requirements.txt，使用pip 进行安装。
   
   ```shell
   pip install -r requirements.txt
   ```

本项目测试环境：香橙派AI pro，CANN 7.0/7.2，python 3.9。

### 模型量化与导出

1. 导出onnx：将transformer库中的modeling_llama替换为export_llama文件下的[modeling_llama](./export_llama/modeling_llama_4.35.py)，通过`python export_llama.py`导出onnx。
2. 模型量化：在export_llama.py中引用量化的配置文件，调用`quantize`函数即可导出量化后的onnx文件，具体见[readme](./export_llama/readme.md)。

### 算子适配

在昇腾安装目录tools/msopgen/template/custom_operator_sample/DSL/Onnx有自定义算子模板，将`matmul_integer_plugin.cc`放在算子模板工程目录下的framework/onnx_plugin下，然后进行编译，安装自定义算子，将Onnx的MatmulInteger对应到昇腾CANN的BatchMatmulV2算子。

### 模型推理运行 

目前提供两种运行模式：
1. cli模式：`cd inference/ && python main.py cli`，在终端运行，每一次输入一行，一次性返回所有的推理结果。
2. web模式：`cd inference/ && python main.py`，前端代码在[github](https://github.com/yinghuo302/ascend-llm-web)或者[gitee](https://gitee.com/yinghuo302/ascend-llm-web)，打包出dist文件夹，放在inference文件夹下即可。

代码需要修改的部分主要在与config.py，可以根据注释修改。inference.py中关于输入格式和结束语判断的部分可能也需要根据具体的模型修改。