import tensorrt as trt

import os
import time

import cv2
import numpy as np
import onnxruntime

from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit

from efficient_vit.demo_sam_model import load_image
import pickle

import sys
sys.path.append('/home/nicole/TensorRT-8.4.3.1/samples/python')
import common


# def predict(batch): # result gets copied into output
#     input_batch = np.array((1, 3, 1280, 720), dtype=np.float32)
#     output = np.empty((1, 3, 1280, 720), dtype = np.float32) 

#     # allocate device memory
#     d_input = cuda.mem_alloc(1 * input_batch.nbytes)
#     d_output = cuda.mem_alloc(1 * output.nbytes)

#     bindings = [int(d_input), int(d_output)]

#     stream = cuda.Stream()
#     # transfer input data to device
#     cuda.memcpy_htod_async(d_input, batch, stream)
#     # execute model
#     context.execute_async_v2(bindings, stream.handle, None)
#     # transfer predictions back
#     cuda.memcpy_dtoh_async(output, d_output, stream)
#     # syncronize threads
#     stream.synchronize()
    
#     return output


if __name__ == '__main__':
    trt.init_libnvinfer_plugins()
    w, h = 1280, 720
    img = load_image("base_imgs/wall.png")
    img = cv2.resize(img, (w, h))

    print("img shape:", img.shape)

    start_point, end_point = (396, 258), (105, 237)


    # # start of engine tests
    f = open('saved_models/updated_yolo.engine', 'rb')
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(f.read())
    # context = engine.create_execution_context()

    # inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    # print("outputs:", len(outputs), outputs)
    # cuda.memcpy_htod_async(inputs[0].device, img, stream)
    # context.execute_async_v2(bindings, stream.handle, None)
    # cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)
    # cuda.memcpy_dtoh_async(outputs[1].host, outputs[1].device, stream)
    # # cuda.memcpy_dtoh_async(outputs[2].host, outputs[2].device, stream)
    # # cuda.memcpy_dtoh_async(outputs[3].host, outputs[3].device, stream)

    # print("\n\noutputs:", outputs)
    # print()


    # for output in outputs:
    #     print(type(output))
    #     print(output)
    #     print()

    # inputs, outputs, bindings = [], [], []
    # stream = cuda.Stream()
    # for binding in engine:
    #     print("binding:", binding)
    #     size = trt.volume(engine.get_binding_shape(binding))
    #     print('binding size:', size)
    #     dtype = trt.nptype(engine.get_binding_dtype(binding))
    #     print('binding dtype:', dtype)
    #     host_mem = cuda.pagelocked_empty(size, dtype)
    #     device_mem = cuda.mem_alloc(host_mem.nbytes)

    #     bindings.append(int(device_mem))
    #     if engine.binding_is_input(binding):
    #         inputs.append({'host': host_mem, 'device': device_mem})
    #     else:
    #         outputs.append({'host': host_mem, 'device': device_mem})

    # inputs[0]['host'] = np.ravel(img)
    # # transfer data to the gpu
    # for inp in inputs:
    #     cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
    # # run inference
    # context.execute_async_v2(
    #     bindings=bindings,
    #     stream_handle=stream.handle)
    # # fetch outputs from gpu
    # for out in outputs:
    #     cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
    # # synchronize stream
    # stream.synchronize()

    # data = [out['host'] for out in outputs]

    # with open("data_out.pickle", "wb") as f:
    #     pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # with open("data_out.pickle", "rb") as f:
    #     data = pickle.load(f)


    # print("data:", data)

    # print("data.length:", len(data))

    # for elt in data:
    #     print(type(elt))
    
    # for elt in data:
    #     if type(elt) != list:
    #         print(elt.shape)
    #     else:
    #         print(len(elt))


    # print('preditions shape:', predictions.shape)




        