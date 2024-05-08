# Llama 模型导出

## 量化

主要参考的量化方案有[LLM.int8](https://arxiv.org/abs/2208.07339)和[SmoothQuant](https://arxiv.org/abs/2211.10438)

量化需要引入`quantize.py`和config文件下的配置文件，目前量化方式共有四种：int8仅权重量化(W8)，int8全量化(W8X8，传入act_scale可以平滑激活)，静态混合精度分解(SD)和动态混合精度分解(W8DX)。根据Llama模型特点，down_proj比qkv_proj, up_proj, gate_proj更难以量化，更深的Decoder Layer更难以量化。配置文件的格式为
```python
quantize_cfg = {
    "0.q_proj":{ # 字典的key为具体某一层，第几个Decoder Layer+线性层名字
        "type":"W8SD", # 量化类型
		"act_scale":True, # type=W8X8表示采用平滑激活，type=W8SD表示用生成的act scale进行静态混合精度分解，如果不采用直接不填这一项，判断时只判断是否存在字典中是否存在act_scale的key-value对，不检查值。
		"alpha":0.85 # 平滑激活的迁移系数，混合精度分解的将多少特征保留为FP16
    },
	"act_scales_path":"/root/zanilia/export_llama/act_scales/llama-2-7b.pt",
	"smooth":True, # SmoothQuant的方案，将激活值的缩放与RMSNorm融合，不会造成额外的开销，但down_proj层无法使用
	"alpha":0.85, #SmoothQuant的迁移系数
}
```
创建新的配置文件方式，新建一个python源文件并提供get函数接口，参数为模型配置和act_scale路径，返回dict格式的量化配置。
在config文件夹下，提供了几个常用的量化配置：int8仅权重量化(w8.py)，int8全量化(w8x8.py)，静态混合精度分解(sd.py)，动态混合精度分解(w8dx.py)，平滑激活(smooth.py)，平滑+静态混合精度分解(smsd.py)。

## 代码解析

### export_llama.py

通过`python export_llama.py`导出onnx

### modeling_llama.py

对llama模型进行修改，主要修改内容
1. 只返回新生成的KV缓存（默认返回所有KV缓存），将返回KV缓存有tuple变为Tensor（torch.cat）
2. 修改LlamaRotaryEmbedding类，原来的方式采用cache的方式，调用时参数为seq_len，如果返回cache的前seq len个元素（如果不足，则需要再次生成）。修改后，调用每次返回max_position_embeddings个元素。所有元素都提前生成了，seq len参数没有使用，不会再次生成，在导出前应将max_position_embeddings设置大一些。
	修改原因主要是：调用LlamaRotaryEmbedding的seq len，为输入长度+kv长度。在apply_rotary_pos_emb，使用position ids为下标取LlamaRotaryEmbedding的输出，获得世纪的PosEmbedding。转om时，这输入长度+kv长度是固定值，如果通过StreamingLLM，H2O等方式驱逐KV缓存，position_ids会超过输入长度+kv长度，导致错误。也可以修改代码获取真实的输入长度+kv长度。

### export_llama.py

将llama模型导出为onnx文件

## quantize.py

量化相关代码，总共有四种方法。

1. W8Linear: int8仅权重量化
2. W8X8Linear: vector-wise absmax int8全量化
3. W8SDLinear: 静态的混合精度分解，分解方式可以使用按权重分解和按act max分解（推荐按act max分解）
4. W8DXLinear: 动态根据输入进行混合精度分解

### smooth.py

直接使用[SmoothQuant](https://github.com/mit-han-lab/smoothquant/) 的相关代码，对激活进行平滑，降低量化难度，调用smooth_lm接口进行量化。smoothquant目前对于q_proj,k_proj,v_proj,gate_proj,up_porj进行平滑，对于down_proj的平滑，可以在W8X8Linear参数中传入act max。

### generate_act_scales.py

直接使用[SmoothQuant](https://github.com/mit-han-lab/smoothquant/) 的相关代码，可以计算某个模型在特定数据集上激活值的最大值，可以用于smoothquant方法的平滑操作和W8SDLinear的混合精度分解。

```bash
python examples/generate_act_scales.py \
    --model-name <model_name_or_path> \
    --output-path <output_act_scales_file_path> \
    --num-samples <num_samples> \
    --seq-len <sequence_length> \
    --dataset-path <path_to_the_calibration_dataset>
```

### change_node.py

将cast fp->int8算子转换为AscendQuant算子，用于atc模型转换


