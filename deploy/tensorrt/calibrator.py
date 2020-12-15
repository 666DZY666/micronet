# *** tensorrt校准模块  ***

import os
import torch
import torch.nn.functional as F
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import ctypes
import logging
import util_trt
logger = logging.getLogger(__name__)
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

# calibrator
class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, input_layers, stream, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)       
        self.input_layers = input_layers
        self.stream = stream
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        self.cache_file = cache_file
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, bindings, names):
        batch = self.stream.next_batch()
        if not batch.size:   
            return None

        cuda.memcpy_htod(self.d_input, batch)
        for i in self.input_layers[0]:
            assert names[0] != i
        
        bindings[0] = int(self.d_input)
        return bindings

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Using calibration cache to save time: {:}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            logger.info("Caching calibration data for future use: {:}".format(self.cache_file))
            f.write(cache)

# calibration_stream
  # mnist/cifar...
class ImageBatchStream():
    def __init__(self, dataset, transform, batch_size, img_size, max_batches):
        self.transform = transform
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.dataset = dataset
          
        self.calibration_data = np.zeros((batch_size,) + img_size, dtype=np.float32) # This is a data holder for the calibration
        self.batch_count = 0
        
    def reset(self):
        self.batch_count = 0
      
    def next_batch(self):
        if self.batch_count < self.max_batches:
            for i in range(self.batch_size): 
                x = self.dataset[i + self.batch_size * self.batch_count]  
                x = util_trt.to_numpy(x).astype(dtype=np.float32)
                if self.transform:
                    x = self.transform(x) 
                self.calibration_data[i] = x.data
            self.batch_count += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32) 
        else:
            return np.array([])
'''
  # ocr
class OCRBatchStream():
    def __init__(self, dataset, transform, batch_size, img_size, max_batches):
        self.transform = transform
        self.batch_size = batch_size
        self.img_size = img_size
        self.max_batches = max_batches
        self.dataset = dataset
        self.args = load_default_param()
          
        self.calibration_data = np.zeros((self.batch_size, *self.img_size), dtype=np.float32) # This is a data holder for the calibration
        self.batch_count = 0
        
    def reset(self):
        self.batch_count = 0
      
    def next_batch(self):
        if self.batch_count < self.max_batches:
            for i in range(self.batch_size):
                x = self.dataset[i + self.batch_size * self.batch_count]['img']
                x = torch.FloatTensor(x)
                x = util_trt.to_numpy(x).astype(dtype=np.float32)
                x = np.transpose(x ,(1, 2, 0))
                # ----------------- resize -----------------
                select_size_list = self.args.select_size_list
                resize_size = select_resize_size(x, select_size_list)
                input_img, ori_scale_img = resize_img(x, resize_size, self.args)
                # ----------------- crop -----------------
                sub_imgs, sub_img_indexes, sub_img_tensors = crop_img_trt(input_img, resize_size, self.args)

                for k in range(len(sub_img_tensors)):
                    if sub_img_tensors[k].shape[2] not in select_size_list or sub_img_tensors[k].shape[3] not in select_size_list:
                        print('size pad error!', sub_img_tensors[k].shape)
                        sys.exit()

                for k in range(len(sub_img_tensors)):
                    if len(sub_img_tensors[k].shape) == 3 and sub_img_tensors[k].shape[0] != 3:
                        sub_img_tensors[k] = np.transpose(sub_img_tensors[k], (2, 0, 1))
                        sub_img_tensors[k] = sub_img_tensors[k][np.newaxis, ...].copy()
                    x = sub_img_tensors[k]
                    # You should implement your own data pipeline for writing the calibration_data   
                    if self.transform:
                        x = self.transform(x) 
                    self.calibration_data[i] = x.data
            self.batch_count += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32) 
        else:
            return np.array([])
'''
  # segmentation
class SegBatchStream():
    def __init__(self, dataset, transform, batch_size, img_size, max_batches):
        self.transform = transform
        self.batch_size = batch_size
        self.img_size = img_size
        self.max_batches = max_batches
        self.dataset = dataset
          
        self.calibration_data = np.zeros((self.batch_size, *self.img_size), dtype=np.float32) # This is a data holder for the calibration
        self.batch_count = 0
        
    def reset(self):
      self.batch_count = 0
      
    def next_batch(self):
        if self.batch_count < self.max_batches:
            for i in range(self.batch_size):
                x = self.dataset[i + self.batch_size * self.batch_count]['img_data'][0]
                x = F.interpolate(x, size=(self.img_size[1], self.img_size[2]))
                x = util_trt.to_numpy(x).astype(dtype=np.float32)
                if self.transform:
                    x = self.transform(x) 
                self.calibration_data[i] = x.data
            self.batch_count += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32) 
        else:
            return np.array([])
            