
import onnx
from onnx import numpy_helper, TensorProto
import numpy as np

output_file = "../../../engines/vit/decoder_k9_fp16_opset18_fixed_size_stacked_output_iou_last_tensorrt_8.6.engine"
onnx_model = onnx.load(output_file)
# Get the graph from the model
graph = onnx_model.graph

print("got graph:", graph)

# Iterate through all nodes in the graph
for node in graph.node:
    if "ReduceMean" in node.op_type:
        print("found a ReduceMean node")
        for index in range(len(node.attribute)):
            if node.attribute[index].name == "axes":
                del node.attribute[index]
                axes_input = onnx.helper.make_tensor_value_info("axes", TensorProto.INT64, [1])
                axes_value = numpy_helper.from_array(np.array([1]), "axes")
                onnx_model.graph.input.extend([axes_input])
                onnx_model.graph.initializer.extend([axes_value])
                node.input.append("axes")
                break