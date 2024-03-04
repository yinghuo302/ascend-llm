'''
Author: zanilia
Date: 2022-12-30 12:56:56
LastEditTime: 2022-12-30 13:11:43
Descripttion: 
'''
from __future__ import absolute_import
import tbe
from te.utils import shape_util
import te.tvm as tvm
import te.lang.cce
import te.platform as tbe_platform
from te.utils import para_check # 提供了通用的算子参数校验接口
from te.utils import shape_util

# 其作用是整网运行时，支持算子做UB自动融合，使得算子在UB中根据UB融合规则自动与其他算子的compute进行拼接，提升算子运行效率，若算子实现逻辑中涉及reshape操作，不可使用此装饰器函数
@tbe.common.register.register_op_compute("Softmax")
def softmax_compute(input_x, output_y, axis=-1, kernel_name="softmax"):
	dtype = input_x.dtype.lower()
	shape = input_x.shape
	has_improve_precision = False #是否需要提升精度
	# 分子：e^x
	data_exp = te.lang.cce.vexp(input_x)
	# 对于输入数据类型为float16的来说，可以先广播到float32，做以下的除法操作，再转为数据类型float16，用于提升计算精度
	tbe_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
	if data_exp.dtype == "float16" and tbe_product in ("Ascend310",):
		data_exp = te.lang.cce.cast_to(data_exp, "float32")
		has_improve_precision = True
	# sum(e^x) 分母，将分母也广播到shape大小
	data_expsum = te.lang.cce.sum(data_exp, axis, keepdims=True)
	data_expsum = te.lang.cce.broadcast(data_expsum, shape)
	# e^x/sum(e^x)
	output = te.lang.cce.vdiv(data_exp, data_expsum)
	#转为数据类型为float16
	if has_improve_precision and dtype == "float16":
		output = te.lang.cce.cast_to(output, "float16")
	return output


# 对算子的输入、输出、属性及Kernel Name进行基础校验
@para_check.check_op_params(para_check.REQUIRED_INPUT,para_check.REQUIRED_OUTPUT,(para_check.OPTION_ATTR_INT,para_check.OPTION_ATTR_LIST_INT), para_check.KERNEL_NAME,para_check.OPTION_ATTR_STR)
def softmax(input_x, output_y, axis=-1, kernel_name="softmax"):
# 获取算子输入tensor的shape以及dtype，为后续定义输入tensor的张量占位符做准备。
	shape = input_x.get("shape")
	dtype = input_x.get("dtype").lower()
	axis=list(axis)
	# 基本校验
	para_check.check_shape(shape,param_name="x")
	para_check.check_dtype(dtype.lower(), ("float16", "float32"),param_name="x")
	axis = shape_util.axis_check(len(shape), axis)
	# 对轴的值进行合法性校验，并返回排好序且是正数的轴值，轴值按照升序进行排序
	shape, axis = shape_util.shape_refine(list(shape), axis)
	# recude dim=1 的轴
	shape, axis = shape_util.simplify_axis_shape(shape, axis) # 把连续的reduce轴进行合并，并把对应的shape的维度也进行合并
	data_x = tvm.placeholder(shape, dtype=dtype, name="data_x")
	with tvm.target.cce():
		output = softmax_compute(data_x, output_y, axis, kernel_name)
		result = tbe.dsl.auto_schedule(output)
	# 调用auto_schedule接口，便可以自动生成算子相应的调度
	tensor_list = [data_x, output]
	# TVM的打印机制；可以看到相应计算的中间表示。配置信息包括是否需要打印IR、是否编译以及算子内核名以及输入、输出张量
	config = {"name": kernel_name,"tensor_list": tensor_list}
	tbe.dsl.build(result, config)