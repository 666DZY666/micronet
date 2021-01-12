# tensorrt-lib

import os
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from calibrator import Calibrator
from torch.autograd import Variable
import torch
import numpy as np
import time
# add verbose
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) # ** engine可视化 **

# create tensorrt-engine
  # fixed and dynamic
def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="",\
               fp16_mode=False, int8_mode=False, calibration_stream=None, calibration_table_path="", save_engine=False, dynamic=False
              ):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(1) as network,\
                trt.OnnxParser(network, TRT_LOGGER) as parser:
            
            # parse onnx model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
                assert network.num_layers > 0, 'Failed to parse ONNX model. \
                            Please check if the ONNX model is compatible '
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))        
            
            # build trt engine
            builder.max_batch_size = max_batch_size
            if not dynamic:
                builder.max_workspace_size = 1 << 30 # 1GB
                builder.fp16_mode = fp16_mode
                if int8_mode:
                    builder.int8_mode = int8_mode
                    assert calibration_stream, 'Error: a calibration_stream should be provided for int8 mode'
                    builder.int8_calibrator  = Calibrator(["input"], calibration_stream, calibration_table_path)
                    print('Int8 mode enabled')
                engine = builder.build_cuda_engine(network) 
            else:
                config = builder.create_builder_config()  
                config.max_workspace_size = 1 << 30  # 1GB
                profile = builder.create_optimization_profile()
                profile.set_shape(network.get_input(0).name, (1, 3, 200, 200), (1, 3, 608, 448), (1, 3, 1200, 1200))
                # dynamic_engine fp16 set
                if fp16_mode: 
                    config.set_flag(trt.BuilderFlag.FP16)
                # dynamic_engine int8 set
                if int8_mode: 
                    config.set_flag(trt.BuilderFlag.INT8)  
                    assert calibration_stream, 'Error: a calibration_stream should be provided for int8 mode'
                    # choose an calibration profile
                    #config.set_calibration_profile(profile)
                    config.int8_calibrator = Calibrator(["input"], calibration_stream, calibration_table_path)
                    print('Int8 mode enabled')
                # choose an optimization profile
                config.add_optimization_profile(profile)
                engine = builder.build_engine(network, config)
            # If errors happend when executing builder.build_cuda_engine(network),
            # a None-Type object would be returned
            if engine is None:
                print('Failed to create the engine')
                return None   
            print("Completed creating the engine")
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine
        
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem 
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# fixed_engine buffer
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# dynamic_engine buffer
def allocate_buffers_v2(engine, h_, w_):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    print('engine.get_binding_format_desc', engine.get_binding_format_desc(0))
    for count,binding in enumerate(engine):
        print('binding:', binding)
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size * h_ * w_
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        print('dtype:', dtype)
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
        print('size:',size)
        print('input:',inputs)
        print('output:',outputs)
        print('------------------')
    return inputs, outputs, bindings, stream

# fixed_engine inference
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # GPU Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    #context.execute(batch_size=batch_size, bindings=bindings)
    # Transfer predictions from GPU to CPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

# dynamic_engine inference
def do_inference_v2(context, bindings, inputs, outputs, stream, h_, w_, binding_id):
    # set the input dimensions
    context.set_binding_shape(binding_id, (1, 3, h_, w_)) 
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    infer_start = time.time()
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle) 
    #context.execute_v2(bindings=bindings)
    infer_end = time.time()
    infer_time = infer_end - infer_start
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs], infer_time

# numpy_array reshape
def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output).copy()
    return h_outputs

# tensor to numpy
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
