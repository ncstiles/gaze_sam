import os
import sys
import logging
import argparse

import numpy as np
import tensorrt as trt
from cuda import cuda, cudart

from load_engine import load_image_encoder_engine


logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

# copying from common.py in branch 3aaa97b
def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))
    
def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res

def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = 100 * (2 ** 30)  # old was 8 GB, but said it wasn't enough memory for all tactics

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        print("network object:", self.network)
        print("explicit precision:", self.network.has_explicit_precision)
        print("num layers:", self.network.num_layers)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info(f"Input '{input.name}' with shape {input.shape} and dtype {input.dtype}")
        
        for output in outputs:
            log.info(f"Output '{output.name}' with shape {output.shape} and dtype {output.dtype}")
        
        assert self.batch_size > 0
        self.builder.max_batch_size = self.batch_size
        log.info(f"Builder max batchsize: {self.batch_size}")

    def create_engine(
        self,
        engine_path,
        encoder=None,
    ):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        """
        print("encoder path:", encoder)
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)

        if not self.builder.platform_has_fast_fp16:
            log.warning("FP16 is not supported natively on this platform/device")
        else:
            self.config.set_flag(trt.BuilderFlag.FP16)
            self.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS) # the three below are equivalent to setting strict_types, but deprecated
            self.config.set_flag(trt.BuilderFlag.DIRECT_IO)
            self.config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)

            print('network num layers in create engine:', self.network.num_layers)

        for i in range(self.network.num_layers):
            layer = self.network.get_layer(i)
            if "norm" in layer.name:
                print("layer contains layerNorm")
                self.network.get_layer(i).precision = trt.DataType.FLOAT
                # for j in range()
            # self.network.get_layer(i).precision = trt.DataType.FLOAT
            # for j in range(self.network.get_layer(i).num_outputs):
            #     self.network.get_layer(i).set_output_type(j
            #     trt.DataType.FLOAT)
            
            # layer = self.network.get_layer(i)
            # if layer.precision != trt.DataType.FLOAT:
            #     print(f"{i+1}. layer type: {layer.type}, name: {layer.name}, precision: {layer.precision}, precision is set: {layer.precision_is_set}")


        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        print("num inputs into the model:", self.network.num_inputs)
        print("inputs:", inputs)


        with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine.serialize())


def main(args):
    builder = EngineBuilder(args.verbose)
    builder.create_network(args.onnx)
    builder.create_engine(
        args.engine,
        args.encoder
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument("--encoder", help="The encoder engine to generate image embeddings")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")

    args = parser.parse_args()
    if not all([args.onnx, args.engine]):
        parser.print_help()
        log.error("These arguments are required: --onnx and --engine")
        sys.exit(1)
    
    main(args)