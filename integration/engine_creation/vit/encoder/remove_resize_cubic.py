import onnx_graphsurgeon as gs
import numpy as np
import onnx
import argparse


parser = argparse.ArgumentParser(
    description="Export the efficient-sam encoder to an onnx model."
)
parser.add_argument(
    "--onnx_path", type=str, default="assets/checkpoints/sam/encoder.onnx",
)

args = parser.parse_args()

graph = gs.import_onnx(onnx.load(args.onnx_path))

for node in graph.nodes:
    if node.op == "Resize":
        print("resizing node", node)
        node.attrs["mode"] = "linear"

# 5. Remove unused nodes/tensors, and topologically sort the graph
# ONNX requires nodes to be topologically sorted to be considered valid.
# Therefore, you should only need to sort the graph when you have added new nodes out-of-order.
# In this case, the identity node is already in the correct spot (it is the last node,
# and was appended to the end of the list), but to be on the safer side, we can sort anyway.
graph.cleanup().toposort()

onnx.save(gs.export_onnx(graph), args.onnx_path)