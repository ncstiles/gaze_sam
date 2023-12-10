from super_gradients.common.object_names import Models
from super_gradients.training import models


import onnx_graphsurgeon as gs 
import onnx
import numpy as np

# does pre- and post- processing
# inputs are UINT8 which are incompatible with trtexec
# https://github.com/Deci-AI/super-gradients/issues/1090
# https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/models_export.md
def create_yolo_onnx(filename): 
    # removed the TensorRT specific backend - leading to onnx validation check failure
    model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
    export_result = model.export(filename)

# TensorRT doesn't support uint8, so cast inputs to floats
# https://forums.developer.nvidia.com/t/exporting-tensorflow-models-to-jetson-nano/154185/10
def convert_inputs_uint8_fp32(convert_from, convert_to):
    graph = gs.import_onnx(onnx.load(convert_from))
    for inp in graph.inputs:
        print("inp:", inp)
        inp.dtype = np.float32

    onnx.save(gs.export_onnx(graph), convert_to)


def is_onnx_model_valid(path):
    model = onnx.load(path)
    try:
        res = onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print("e:", e)
        return False
    return True



if __name__ == '__main__':
    onnx_generated_file = 'yolo_nas_s_v2.onnx'
    create_yolo_onnx(onnx_generated_file)
    print("onnx file valid?", is_onnx_model_valid(onnx_generated_file))

    no_uint8_onnx_file = 'no_uint8_yolo_nas_s_v2.onnx'
    convert_inputs_uint8_fp32(onnx_generated_file, no_uint8_onnx_file)
    print("onnx file valid?", is_onnx_model_valid(no_uint8_onnx_file))

    # print(is_onnx_model_valid('no_uint8_onnx_compatible_yolo_nas_s_v2.onnx'))

# Output from ONNX and engine building for future reference

# The console stream is logged into /home/nicole/sg_logs/console.log
# [2023-10-14 17:07:32] INFO - crash_tips_setup.py - Crash tips is enabled. You can set your environment variable to CRASH_HANDLER=FALSE to disable it
# [2023-10-14 17:07:32] WARNING - __init__.py - Failed to import pytorch_quantization
# /home/nicole/miniconda3/envs/efficientvit/lib/python3.10/site-packages/_distutils_hack/__init__.py:33: UserWarning: Setuptools is replacing distutils.
#   warnings.warn("Setuptools is replacing distutils.")
# [2023-10-14 17:07:46] WARNING - calibrator.py - Failed to import pytorch_quantization
# [2023-10-14 17:07:46] WARNING - export.py - Failed to import pytorch_quantization
# [2023-10-14 17:07:46] WARNING - selective_quantization_utils.py - Failed to import pytorch_quantization
# [2023-10-14 17:07:47] INFO - checkpoint_utils.py - License Notification: YOLO-NAS pre-trained weights are subjected to the specific license terms and conditions detailed in 
# https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md
# By downloading the pre-trained weight files you agree to comply with these terms.
# [2023-10-14 17:07:47] INFO - checkpoint_utils.py - Successfully loaded pretrained weights for architecture yolo_nas_s
# ============= Diagnostic Run torch.onnx.export version 2.0.1+cu117 =============
# verbose: False, log level: Level.ERROR
# ======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================

# [2023-10-14 17:07:55] INFO - nms.py - Created NMS plugin 'EfficientNMS_TRT' with attributes: {'plugin_version': '1', 'background_class': -1, 'max_output_boxes': 1000, 'score_threshold': 0.25, 'iou_threshold': 0.7, 'score_activation': False, 'box_coding': 0}
# Model exported successfully to yolo_nas_s.onnx
# Model expects input image of shape [1, 3, 640, 640]
# Input image dtype is torch.uint8
# Exported model already contains preprocessing (normalization) step, so you don't need to do it manually.
# Preprocessing steps to be applied to input image are:
# Sequential(
#   (0): CastTensorTo(dtype=torch.float32)
#   (1): ApplyMeanStd(mean=[0.], scale=[255.])
# )

# Exported model contains postprocessing (NMS) step with the following parameters:
#     num_pre_nms_predictions=1000
#     max_predictions_per_image=1000
#     nms_threshold=0.7
#     confidence_threshold=0.25
#     output_predictions_format=batch

# Exported model is in ONNX format and can be used with TensorRT
# To run inference with TensorRT, please see TensorRT deployment documentation
# You can benchmark the model using the following code snippet:

#     trtexec --onnx=yolo_nas_s.onnx --fp16 --avgRuns=100 --duration=15

# Exported model has predictions in batch format:

#     num_detections, pred_boxes, pred_scores, pred_classes = predictions
#     for image_index in range(num_detections.shape[0]):
#       for i in range(num_detections[image_index,0]):
#         class_id = pred_classes[image_index, i]
#         confidence = pred_scores[image_index, i]
#         x_min, y_min, x_max, y_max = pred_boxes[image_index, i]
#         print(f"Detected object with class_id={class_id}, confidence={confidence}, x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
# (efficientvit) nicole@g0:~/gaze_sam$ 




# _____







# (efficientvit) nicole@g0:~/gaze_sam$ trtexec --onnx=updated_yolo_model.onnx --saveEngine=updated_yolo_model.engine
# &&&& RUNNING TensorRT.trtexec [TensorRT v8403] # /home/nicole/TensorRT-8.4.3.1/bin/trtexec --onnx=updated_yolo_model.onnx --saveEngine=updated_yolo_model.engine
# [10/14/2023-22:29:21] [I] === Model Options ===
# [10/14/2023-22:29:21] [I] Format: ONNX
# [10/14/2023-22:29:21] [I] Model: updated_yolo_model.onnx
# [10/14/2023-22:29:21] [I] Output:
# [10/14/2023-22:29:21] [I] === Build Options ===
# [10/14/2023-22:29:21] [I] Max batch: explicit batch
# [10/14/2023-22:29:21] [I] Memory Pools: workspace: default, dlaSRAM: default, dlaLocalDRAM: default, dlaGlobalDRAM: default
# [10/14/2023-22:29:21] [I] minTiming: 1
# [10/14/2023-22:29:21] [I] avgTiming: 8
# [10/14/2023-22:29:21] [I] Precision: FP32
# [10/14/2023-22:29:21] [I] LayerPrecisions: 
# [10/14/2023-22:29:21] [I] Calibration: 
# [10/14/2023-22:29:21] [I] Refit: Disabled
# [10/14/2023-22:29:21] [I] Sparsity: Disabled
# [10/14/2023-22:29:21] [I] Safe mode: Disabled
# [10/14/2023-22:29:21] [I] DirectIO mode: Disabled
# [10/14/2023-22:29:21] [I] Restricted mode: Disabled
# [10/14/2023-22:29:21] [I] Build only: Disabled
# [10/14/2023-22:29:21] [I] Save engine: updated_yolo_model.engine
# [10/14/2023-22:29:21] [I] Load engine: 
# [10/14/2023-22:29:21] [I] Profiling verbosity: 0
# [10/14/2023-22:29:21] [I] Tactic sources: Using default tactic sources
# [10/14/2023-22:29:21] [I] timingCacheMode: local
# [10/14/2023-22:29:21] [I] timingCacheFile: 
# [10/14/2023-22:29:21] [I] Input(s)s format: fp32:CHW
# [10/14/2023-22:29:21] [I] Output(s)s format: fp32:CHW
# [10/14/2023-22:29:21] [I] Input build shapes: model
# [10/14/2023-22:29:21] [I] Input calibration shapes: model
# [10/14/2023-22:29:21] [I] === System Options ===
# [10/14/2023-22:29:21] [I] Device: 0
# [10/14/2023-22:29:21] [I] DLACore: 
# [10/14/2023-22:29:21] [I] Plugins:
# [10/14/2023-22:29:21] [I] === Inference Options ===
# [10/14/2023-22:29:21] [I] Batch: Explicit
# [10/14/2023-22:29:21] [I] Input inference shapes: model
# [10/14/2023-22:29:21] [I] Iterations: 10
# [10/14/2023-22:29:21] [I] Duration: 3s (+ 200ms warm up)
# [10/14/2023-22:29:21] [I] Sleep time: 0ms
# [10/14/2023-22:29:21] [I] Idle time: 0ms
# [10/14/2023-22:29:21] [I] Streams: 1
# [10/14/2023-22:29:21] [I] ExposeDMA: Disabled
# [10/14/2023-22:29:21] [I] Data transfers: Enabled
# [10/14/2023-22:29:21] [I] Spin-wait: Disabled
# [10/14/2023-22:29:21] [I] Multithreading: Disabled
# [10/14/2023-22:29:21] [I] CUDA Graph: Disabled
# [10/14/2023-22:29:21] [I] Separate profiling: Disabled
# [10/14/2023-22:29:21] [I] Time Deserialize: Disabled
# [10/14/2023-22:29:21] [I] Time Refit: Disabled
# [10/14/2023-22:29:21] [I] Inputs:
# [10/14/2023-22:29:21] [I] === Reporting Options ===
# [10/14/2023-22:29:21] [I] Verbose: Disabled
# [10/14/2023-22:29:21] [I] Averages: 10 inferences
# [10/14/2023-22:29:21] [I] Percentile: 99
# [10/14/2023-22:29:21] [I] Dump refittable layers:Disabled
# [10/14/2023-22:29:21] [I] Dump output: Disabled
# [10/14/2023-22:29:21] [I] Profile: Disabled
# [10/14/2023-22:29:21] [I] Export timing to JSON file: 
# [10/14/2023-22:29:21] [I] Export output to JSON file: 
# [10/14/2023-22:29:21] [I] Export profile to JSON file: 
# [10/14/2023-22:29:21] [I] 
# [10/14/2023-22:29:24] [I] === Device Information ===
# [10/14/2023-22:29:24] [I] Selected Device: NVIDIA GeForce GTX 1080 Ti
# [10/14/2023-22:29:24] [I] Compute Capability: 6.1
# [10/14/2023-22:29:24] [I] SMs: 28
# [10/14/2023-22:29:24] [I] Compute Clock Rate: 1.62 GHz
# [10/14/2023-22:29:24] [I] Device Global Memory: 11178 MiB
# [10/14/2023-22:29:24] [I] Shared Memory per SM: 96 KiB
# [10/14/2023-22:29:24] [I] Memory Bus Width: 352 bits (ECC disabled)
# [10/14/2023-22:29:24] [I] Memory Clock Rate: 5.505 GHz
# [10/14/2023-22:29:24] [I] 
# [10/14/2023-22:29:24] [I] TensorRT version: 8.4.3
# [10/14/2023-22:29:25] [I] [TRT] [MemUsageChange] Init CUDA: CPU +194, GPU +0, now: CPU 203, GPU 229 (MiB)
# [10/14/2023-22:29:26] [I] [TRT] [MemUsageChange] Init builder kernel library: CPU +6, GPU +2, now: CPU 228, GPU 231 (MiB)
# [10/14/2023-22:29:26] [I] Start parsing network model
# [10/14/2023-22:29:26] [I] [TRT] ----------------------------------------------------------------
# [10/14/2023-22:29:26] [I] [TRT] Input filename:   updated_yolo_model.onnx
# [10/14/2023-22:29:26] [I] [TRT] ONNX IR version:  0.0.7
# [10/14/2023-22:29:26] [I] [TRT] Opset version:    14
# [10/14/2023-22:29:26] [I] [TRT] Producer name:    
# [10/14/2023-22:29:26] [I] [TRT] Producer version: 
# [10/14/2023-22:29:26] [I] [TRT] Domain:           
# [10/14/2023-22:29:26] [I] [TRT] Model version:    0
# [10/14/2023-22:29:26] [I] [TRT] Doc string:       
# [10/14/2023-22:29:26] [I] [TRT] ----------------------------------------------------------------
# [10/14/2023-22:29:26] [W] [TRT] onnx2trt_utils.cpp:369: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
# [10/14/2023-22:29:26] [I] [TRT] No importer registered for op: EfficientNMS_TRT. Attempting to import as plugin.
# [10/14/2023-22:29:26] [I] [TRT] Searching for plugin: EfficientNMS_TRT, plugin_version: 1, plugin_namespace: 
# [10/14/2023-22:29:26] [I] [TRT] Successfully created plugin: EfficientNMS_TRT
# [10/14/2023-22:29:26] [I] Finish parsing network model
# [10/14/2023-22:29:28] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +267, GPU +110, now: CPU 552, GPU 341 (MiB)
# [10/14/2023-22:29:28] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +113, GPU +48, now: CPU 665, GPU 389 (MiB)
# [10/14/2023-22:29:28] [I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
# [10/14/2023-22:31:13] [I] [TRT] Detected 1 inputs and 4 output network tensors.
# [10/14/2023-22:31:14] [I] [TRT] Total Host Persistent Memory: 171856
# [10/14/2023-22:31:14] [I] [TRT] Total Device Persistent Memory: 2021376
# [10/14/2023-22:31:14] [I] [TRT] Total Scratch Memory: 3072000
# [10/14/2023-22:31:14] [I] [TRT] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 0 MiB, GPU 0 MiB
# [10/14/2023-22:31:14] [I] [TRT] [BlockAssignment] Algorithm ShiftNTopDown took 32.5469ms to assign 9 blocks to 138 nodes requiring 41292800 bytes.
# [10/14/2023-22:31:14] [I] [TRT] Total Activation Memory: 41292800
# [10/14/2023-22:31:14] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 986, GPU 575 (MiB)
# [10/14/2023-22:31:14] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +1, GPU +10, now: CPU 987, GPU 585 (MiB)
# [10/14/2023-22:31:14] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
# [10/14/2023-22:31:14] [W] [TRT] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
# [10/14/2023-22:31:14] [W] [TRT] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
# [10/14/2023-22:31:16] [I] Engine built in 111.4 sec.
# [10/14/2023-22:31:16] [I] [TRT] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 975, GPU 495 (MiB)
# [10/14/2023-22:31:16] [I] [TRT] Loaded engine size: 55 MiB
# [10/14/2023-22:31:16] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +10, now: CPU 979, GPU 561 (MiB)
# [10/14/2023-22:31:16] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +1, GPU +8, now: CPU 980, GPU 569 (MiB)
# [10/14/2023-22:31:16] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
# [10/14/2023-22:31:16] [I] Engine deserialized in 0.0503549 sec.
# [10/14/2023-22:31:16] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 980, GPU 563 (MiB)
# [10/14/2023-22:31:16] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 980, GPU 571 (MiB)
# [10/14/2023-22:31:16] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
# [10/14/2023-22:31:16] [I] Using random values for input onnx::Cast_0
# [10/14/2023-22:31:16] [I] Created input binding for onnx::Cast_0 with dimensions 1x3x640x640
# [10/14/2023-22:31:16] [I] Using random values for output num_dets
# [10/14/2023-22:31:16] [I] Created output binding for num_dets with dimensions 1x1
# [10/14/2023-22:31:16] [I] Using random values for output det_boxes
# [10/14/2023-22:31:16] [I] Created output binding for det_boxes with dimensions 1x1000x4
# [10/14/2023-22:31:16] [I] Using random values for output det_scores
# [10/14/2023-22:31:16] [I] Created output binding for det_scores with dimensions 1x1000
# [10/14/2023-22:31:16] [I] Using random values for output det_classes
# [10/14/2023-22:31:16] [I] Created output binding for det_classes with dimensions 1x1000
# [10/14/2023-22:31:16] [I] Starting inference
# [10/14/2023-22:31:19] [I] Warmup completed 37 queries over 200 ms
# [10/14/2023-22:31:19] [I] Timing trace has 546 queries over 3.0151 s
# [10/14/2023-22:31:19] [I] 
# [10/14/2023-22:31:19] [I] === Trace details ===
# [10/14/2023-22:31:19] [I] Trace averages of 10 runs:
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.45834 ms - Host latency: 5.9307 ms (enqueue 4.10819 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46724 ms - Host latency: 5.94354 ms (enqueue 3.48126 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.45609 ms - Host latency: 5.93575 ms (enqueue 2.92395 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46193 ms - Host latency: 5.94755 ms (enqueue 2.94477 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46807 ms - Host latency: 5.95695 ms (enqueue 2.9399 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46225 ms - Host latency: 5.9515 ms (enqueue 2.94243 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.47166 ms - Host latency: 5.95733 ms (enqueue 2.9549 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.44844 ms - Host latency: 5.93717 ms (enqueue 2.96566 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.68874 ms - Host latency: 6.15851 ms (enqueue 4.5933 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.45917 ms - Host latency: 5.94224 ms (enqueue 2.79877 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.47912 ms - Host latency: 5.96899 ms (enqueue 2.93807 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46171 ms - Host latency: 5.95167 ms (enqueue 3.48039 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.45076 ms - Host latency: 5.92584 ms (enqueue 3.72765 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 7.3182 ms - Host latency: 7.79614 ms (enqueue 5.1692 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.48539 ms - Host latency: 5.9684 ms (enqueue 2.91318 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.4778 ms - Host latency: 5.96173 ms (enqueue 3.82131 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46677 ms - Host latency: 5.94526 ms (enqueue 3.84316 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46223 ms - Host latency: 5.93739 ms (enqueue 3.74907 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.47278 ms - Host latency: 5.95417 ms (enqueue 3.70369 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.45929 ms - Host latency: 5.93643 ms (enqueue 3.20894 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46603 ms - Host latency: 5.94547 ms (enqueue 3.24319 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.45559 ms - Host latency: 5.93816 ms (enqueue 3.18231 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46736 ms - Host latency: 5.95065 ms (enqueue 3.71693 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.47075 ms - Host latency: 5.9499 ms (enqueue 3.70024 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.47135 ms - Host latency: 5.94829 ms (enqueue 3.73717 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.4684 ms - Host latency: 5.94492 ms (enqueue 3.69419 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.48928 ms - Host latency: 5.96016 ms (enqueue 3.65579 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.49152 ms - Host latency: 5.96692 ms (enqueue 3.3447 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.47163 ms - Host latency: 5.95259 ms (enqueue 3.26959 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.44728 ms - Host latency: 5.92502 ms (enqueue 3.3765 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46923 ms - Host latency: 5.94955 ms (enqueue 3.19623 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46775 ms - Host latency: 5.94712 ms (enqueue 2.842 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46067 ms - Host latency: 5.94121 ms (enqueue 2.79763 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.48529 ms - Host latency: 5.95541 ms (enqueue 3.7939 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46304 ms - Host latency: 5.93733 ms (enqueue 3.64524 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46912 ms - Host latency: 5.93943 ms (enqueue 3.2428 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46501 ms - Host latency: 5.94717 ms (enqueue 3.2668 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46521 ms - Host latency: 5.94758 ms (enqueue 3.77314 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.45095 ms - Host latency: 5.9291 ms (enqueue 2.87085 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.4603 ms - Host latency: 5.94053 ms (enqueue 3.24321 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.4542 ms - Host latency: 5.9321 ms (enqueue 3.26826 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.4791 ms - Host latency: 5.96255 ms (enqueue 3.23687 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.49619 ms - Host latency: 5.96821 ms (enqueue 3.38123 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46416 ms - Host latency: 5.94517 ms (enqueue 3.1927 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46458 ms - Host latency: 5.94407 ms (enqueue 3.67346 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.47097 ms - Host latency: 5.9543 ms (enqueue 3.75303 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46658 ms - Host latency: 5.94595 ms (enqueue 3.24456 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.45435 ms - Host latency: 5.93396 ms (enqueue 3.79297 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46426 ms - Host latency: 5.95107 ms (enqueue 3.21951 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.47195 ms - Host latency: 5.95254 ms (enqueue 3.20156 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46375 ms - Host latency: 5.94275 ms (enqueue 3.21233 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.45481 ms - Host latency: 5.93259 ms (enqueue 3.20439 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46572 ms - Host latency: 5.94565 ms (enqueue 3.51013 ms)
# [10/14/2023-22:31:19] [I] Average on 10 runs - GPU latency: 5.46304 ms - Host latency: 5.94524 ms (enqueue 2.86433 ms)
# [10/14/2023-22:31:19] [I] 
# [10/14/2023-22:31:19] [I] === Performance summary ===
# [10/14/2023-22:31:19] [I] Throughput: 181.088 qps
# [10/14/2023-22:31:19] [I] Latency: min = 5.79297 ms, max = 24.5749 ms, mean = 5.98386 ms, median = 5.94467 ms, percentile(99%) = 6.12769 ms
# [10/14/2023-22:31:19] [I] Enqueue Time: min = 1.87024 ms, max = 25.7644 ms, mean = 3.3925 ms, median = 2.92749 ms, percentile(99%) = 8.42065 ms
# [10/14/2023-22:31:19] [I] H2D Latency: min = 0.411774 ms, max = 0.493347 ms, mean = 0.461326 ms, median = 0.464874 ms, percentile(99%) = 0.479858 ms
# [10/14/2023-22:31:19] [I] GPU Compute Time: min = 5.33301 ms, max = 24.0527 ms, mean = 5.5041 ms, median = 5.46616 ms, percentile(99%) = 5.64722 ms
# [10/14/2023-22:31:19] [I] D2H Latency: min = 0.00805664 ms, max = 0.0578613 ms, mean = 0.0184306 ms, median = 0.0178223 ms, percentile(99%) = 0.026123 ms
# [10/14/2023-22:31:19] [I] Total Host Walltime: 3.0151 s
# [10/14/2023-22:31:19] [I] Total GPU Compute Time: 3.00524 s
# [10/14/2023-22:31:19] [W] * GPU compute time is unstable, with coefficient of variance = 14.5636%.
# [10/14/2023-22:31:19] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
# [10/14/2023-22:31:19] [I] Explanations of the performance metrics are printed in the verbose logs.
# [10/14/2023-22:31:19] [I] 
# &&&& PASSED TensorRT.trtexec [TensorRT v8403] # /home/nicole/TensorRT-8.4.3.1/bin/trtexec --onnx=updated_yolo_model.onnx --saveEngine=updated_yolo_model.engine
# (efficientvit) nicole@g0:~/gaze_sam$ 
