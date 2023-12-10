from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.conversion import ExportTargetBackend

import onnx_graphsurgeon as gs
import numpy as np
import onnx

import argparse

parser = argparse.ArgumentParser("onnxruntime demo")
parser.add_argument("--model-name", default="yolo.onnx", type=str)
args = parser.parse_args()

# create base model. important to use the tensorRT backend or will run into issues converting to trt (padding node resize error)
model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
export_result = model.export(args.model_name, engine=ExportTargetBackend.TENSORRT)


# change type of input node from uint8 to fp32 (trt doesn't support uint8)
# graph.nodes is a list, the inputs are also list.  you can get the input (type variable), use the __dict__ to see its attributes
graph = gs.import_onnx(onnx.load(args.model_name))

node_to_replace = graph.nodes[0]
node_to_replace.inputs[0].dtype = 1 # convert the input node from uint8 to fp32
onnx_model = gs.export_onnx(graph)
onnx.save(onnx_model, args.model_name)