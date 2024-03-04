/*
 * @Author: zanilia
 * @Date: 2022-12-30 12:55:08
 * @LastEditTime: 2022-12-30 12:57:50
 * @Descripttion: 
 */
#include "./softmax.h"
#include <string>
#include <vector>
namespace ge
{
	//----------------Softmax-------------------
	// 实现Verify方法
	IMPLEMT_VERIFIER(Softmax, SoftmaxVerify)
	{
		return GRAPH_SUCCESS;
	}
	// 实现InferShape方法，描述tensor输入输出信息
	// IMPLEMT_COMMON_INFERFUNC(func_name)：自动生成一个类型为Operator类的对象op
	// 可直接调用Operator类接口进行InferShape的实现。其中，func_name：用户自定义
	IMPLEMT_COMMON_INFERFUNC(SoftmaxInferShape)
	{
		TensorDesc tensordesc_output = op.GetOutputDesc("y");
		tensordesc_output.SetShape(op.GetInputDesc("x").GetShape());	   // 更新输出的shape
		tensordesc_output.SetDataType(op.GetInputDesc("x").GetDataType()); // 更新输出的type
		tensordesc_output.SetFormat(op.GetInputDesc("x").GetFormat());	   // 更新输出的格式
		(void)op.UpdateOutputDesc("y", tensordesc_output);
		return GRAPH_SUCCESS;
	}
	// 注册算子的InferShape函数
	COMMON_INFER_FUNC_REG(Softmax, SoftmaxInferShape);
	// 注册算子的Verify函数
	VERIFY_FUNC_REG(Softmax, SoftmaxVerify);
}
