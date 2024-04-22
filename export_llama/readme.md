#### modeling_llama.py

对llama模型进行修改，主要修改内容
1. 只返回新生成的KV缓存（默认返回所有KV缓存），将返回KV缓存有tuple变为Tensor（torch.cat）
2. 修改LlamaRotaryEmbedding类，原来的方式采用cache的方式，调用时参数为seq_len，如果返回cache的前seq len个元素（如果不足，则需要再次生成）。修改后，调用每次返回max_position_embeddings个元素。所有元素都提前生成了，seq len参数没有使用，不会再次生成，在导出前应将max_position_embeddings设置大一些。
	修改原因主要是：调用LlamaRotaryEmbedding的seq len，为输入长度+kv长度。在apply_rotary_pos_emb，使用position ids为下标取LlamaRotaryEmbedding的输出，获得世纪的PosEmbedding。转om时，这输入长度+kv长度是固定值，如果通过streamllm，H2O等方式驱逐KV缓存，position_ids会超过输入长度+kv长度，导致错误。也可以修改代码获取真实的输入长度+kv长度。

### export_llama.py

将llama模型导出为onnx文件

## quantize.py

量化相关代码，总共有四种方法。
1. W8Linear: int8仅权重量化
2. W8X8Linear: vector-wise absmax int8全量化
3. W8SDLinear: 静态的混合精度分解，分解方式可以使用按权重分解和按act max分解（推荐）
4. W8DXLinear: 动态根据输入进行混合精度分解
config文件下四个配置文件分别与上面四种量化方法对应。量化时调用quantize函数，传入配置文件中的config即可。对于llama-7B建议，使用混合精度分解；对于tiny-llama-1.1B模型，直接使用W8X8Linear即可。

### smooth.py

直接使用https://github.com/mit-han-lab/smoothquant/ 的相关代码，对激活进行平滑，降低量化难度。smoothquant目前对于q_proj,k_proj,v_proj,gate_proj,up_porj进行平滑，对于down_proj的平滑，可以在W8X8Linear参数中传入act max。

### generate_act_scales.py

直接使用https://github.com/mit-han-lab/smoothquant/ 的相关代码，可以计算某个模型在特定数据集上激活值的最大值，可以用于smoothquant方法的平滑操作和W8SDLinear的混合精度分解。

### change_node.py

将cast fp->int8算子转换为AscendQuant算子，用于atc模型转换


