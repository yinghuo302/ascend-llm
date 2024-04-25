import argparse
import onnx
import onnx.helper as helper
from onnx import TensorProto
def change_node(in_path,out_path):
	model = onnx.load(in_path)
	new_nodes = []

	for node in model.graph.node:
		# 判断节点类型
		new_node = node
		if node.op_type == "Cast":
			# 替换为新的算子类型, 昇腾Cast fp16 -> int8 有精度问题，暂时用AscendQuant
			to_attribute = next(attr for attr in node.attribute if attr.name == "to")
			if to_attribute.i == TensorProto.INT8:
				new_node = helper.make_node(
					"AscendQuant",
					inputs=node.input,
					outputs=node.output,
					offset=0.,
					scale=1.,
				)
		new_nodes.append(new_node)

	new_graph = helper.make_graph(
		new_nodes,
		"new_graph",
		inputs=model.graph.input,
		outputs=model.graph.output,
		value_info=model.graph.value_info,
		initializer=model.graph.initializer
	)

	new_model = helper.make_model(new_graph, producer_name=model.producer_name,opset_imports=model.opset_import,ir_version = model.ir_version)
	# new_model.ir_version = model.ir_version
	# new_model.opset_import = model.opset_import
	# new_model.metadata_props = model.metadata_props
	onnx.save(new_model, out_path,save_as_external_data=True,size_threshold=0,convert_attribute=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", 
		type=str,
		default="./model/export_out/tiny-llama.onnx",
		help="path to onnx model that need to be processed"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./model/change_node_out/tiny-llama.onnx",
        help="where to save new onnx model",
    )
    args = parser.parse_args()
    change_node(args.input,args.output)