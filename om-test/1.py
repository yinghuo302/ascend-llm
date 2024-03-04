import onnxruntime

# 加载模型
onnx_model_path = "/root/program/test_quant.onnx"


options = onnxruntime.SessionOptions()

providers = [
    (
        "CANNExecutionProvider",
        {
            "device_id": 0,
            # "arena_extend_strategy": "kNextPowerOfTwo",
            "npu_mem_limit": 2 * 1024 * 1024 * 1024,
            "op_select_impl_mode": "high_performance",
            # "optypelist_for_implmode": "Gelu",
            "enable_cann_graph": True
        },
    ),
    "CPUExecutionProvider",
]
sess = onnxruntime.InferenceSession(onnx_model_path,sess_options=options,providers=providers)
import numpy as np
# 准备输入数据

# 执行推理
import time
for i in range(10):
	begin_time = time.time()
	input_data = np.random.randn(i, 1000).astype(np.float32)  # 示例数据，注意替换成您的实际数据
	output = sess.run(None, {'x.1': input_data})  # 替换 'input_name' 为您模型中输入的名称
	print(f"use {time.time()-begin_time}s")
	print(output)
# decorator,attrs,psutil,absl-py,cloudpickle,synr==0.5.0,tornado