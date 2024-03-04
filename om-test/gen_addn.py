import os
import numpy as np
import onnx

def gen_onnx():
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [5])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [5])
    Z = onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [5])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [5])

    node0 = onnx.helper.make_node("AddN", inputs=["X", "Y", "Z"], outputs=["output"])

    inputs = [X, Y, Z]
    outputs = [output]

    graph_def = onnx.helper.make_graph(
        [node0],
        "addn_model",
        inputs,
        outputs
    )

    model_def = onnx.helper.make_model(graph_def)
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "addn_model.onnx")
    print(model_def)

if __name__ == "__main__":
    gen_onnx()