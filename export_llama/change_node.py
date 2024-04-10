import onnx
import onnx.helper as helper
from onnx import TensorProto

model = onnx.load("/home/zanilia/program/AI/models/llama-onnx/llama.onnx")
new_nodes = []

for node in model.graph.node:
    # 判断节点类型
    new_node = node
    if node.op_type == "Cast":
        # 替换为新的算子类型
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
onnx.save(new_model, "/home/zanilia/program/AI/models/llama_1.onnx")