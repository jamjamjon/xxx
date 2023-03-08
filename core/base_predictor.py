import tensorrt as trt
import numpy as np
import onnxruntime
from pathlib import Path
import cv2
import yaml
from omegaconf import DictConfig, OmegaConf
import hydra
from abc import ABC, abstractmethod
import rich
from rich import inspect
from rich.live import Live
import sys
from flask import Flask, request, jsonify
import weakref
import inspect
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from collections import OrderedDict
# from numba import jit
import ctypes
import torch

from collections import namedtuple
import pycuda.autoinit  # noqa F401
import pycuda.driver as cuda
# import tensorrt as trt
from typing import List, Optional, Tuple, Union


# path
FILE = Path(__file__).resolve() # file resolve path
ROOT_DIR = FILE.parents[1]  # ROOT_dir
PARENT_DIR = FILE.parents[0]    # parent dir 


# add to python path
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR)) 


# modules
from core.utils import (WEIGHTS_TYPE, MODEL_TYPE, IMG_FORMAT, CenterCrop, Softmax, Normalize, letterbox, 
                        parse_model_config, CONSOLE, MB, GB, KB, pysizeof, LOGGER, INSPECTOR,
                        scale_boxes_batch, xywh2xyxy, TIMER, batched_nms, gpu_info, resource_info,
                        InstanceInfo, DeviceInfo)


# yaml path (modify this for every project)
# CONFIG_DIR = str(ROOT_DIR / "deploy/projects/playing_phone/configs")
CONFIG_DIR = str(ROOT_DIR / "deploy/projects/mask_wearing/configs")
CONFIG_NAME = "default"

# ----------------------------------------------------------------------------------------------------------------


class Predictor:
    # base infer model

    def __init__(self, config, 
                    device: DeviceInfo,  # new 
                    do_warmup=False 
                    # fp16=False
                    # , device='cpu'   # TODO: device, half, ..
                ):
        # self._custom_methods_list = self._get_custom_methods()  # custom methods

        # load config
        assert config is not None, f"{self.__class__.__name__} has no config!"
        config = config[0] if isinstance(config, list) else config   # now only catch first list item
        self.__dict__.update(config)

        # device
        self.device = device  # override device

        # check weights type
        self.onnx, self.pt, self.engine = False, False, False
        if self.model_weights is not None:
            suffix = Path(self.model_weights).suffix.lower()
            assert suffix in WEIGHTS_TYPE, f'{suffixx} is not supported.'
        for x, y in zip(WEIGHTS_TYPE, (suffix == x for x in WEIGHTS_TYPE)):
            setattr(self, x[1:], y)

        # check model type
        assert getattr(self, 'model_type', None), f"`model_type` in config.yaml is not named!"
        self.is_detector, self.is_classifier, self.is_other = (self.model_type.lower() == x for x in MODEL_TYPE)

        # device shifted
        self.is_device_shifted = False


        # multi backends setup
        if self.onnx:   # onnx

            # create Runtime session
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': self.device.id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                    # 'gpu_mem_limit': 5 * GB,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider'
            ] if self.device.type == 'cuda' else ['CPUExecutionProvider']
    
            # session_options = onnxruntime.SessionOptions()
            # session_options.enable_profiling=True
            self.session = onnxruntime.InferenceSession(self.model_weights, providers=providers)  # create session
            self.io_binding = self.session.io_binding()  # io_binding 
            self.input_names = [x.name for x in self.session.get_inputs()]
            self.input_shape = self.session.get_inputs()[0].shape
            self.input_size = self.input_shape[-2:]
            self.output_names = [x.name for x in self.session.get_outputs()]
            self.output_shape = self.session.get_outputs()[0].shape
            
            # get onnx meta
            meta = self.session.get_modelmeta().custom_metadata_map  
            self.classes_names = eval(meta['names']) if 'names' in meta else None

            # init device mem
            self.ort_x, self.ort_y = None, None

        elif self.pt:  # Not Now
            pass

        elif self.engine: # TODO
            LOGGER.info(f"TRT: {trt.__version__}")

            # self.ctx = cuda.Device(self.device.id).make_context()

            logger = trt.Logger(trt.Logger.INFO)
            with open(self.model_weights, 'rb') as f, trt.Runtime(logger) as runtime:
                self.model = runtime.deserialize_cuda_engine(f.read())
            self.context = self.model.create_execution_context()
            self.inputs, self.outputs, self.bindings = [], [], []
            self.stream = cuda.Stream()
            self.dynamic, self.fp16, self.fp32, self.int8 = False, False, False, False
            self.num_inputs = np.sum(self.model.binding_is_input(i) for i in range(self.model.num_bindings))
            self.num_outputs = self.model.num_bindings - self.num_inputs

            for i in range(self.model.num_bindings):

                dtype = trt.nptype(self.model.get_binding_dtype(i))  # numpy dtype
                name = self.model.get_binding_name(i)   
                shape = self.model.get_binding_shape(i) 

                # info
                rich.print(f"> {i} | {name} | {shape} | {dtype} | context shape: {self.context.get_binding_shape(i)}")

                # fp16 / fp32 / int8
                if dtype == np.float16:
                    self.fp16 = True
                elif dtype == np.float32:
                    self.fp32 = True
                elif dtype == np.int8:
                    self.int8 = True


                # inputs
                if self.model.binding_is_input(i):
                    # dynamic shape
                    if -1 in tuple(self.model.get_binding_shape(i)):
                        self.dynamic = True
                        self.dynamic_min_shape = tuple(self.model.get_profile_shape(0, i)[0])
                        self.dynamic_opt_shape = tuple(self.model.get_profile_shape(0, i)[1])
                        self.dynamic_max_shape = tuple(self.model.get_profile_shape(0, i)[2])
                        self.context.set_binding_shape(i, self.dynamic_max_shape)   # max dynamic shape
                        
                        host_mem = np.empty(self.dynamic_max_shape, dtype)
                        device_mem = cuda.mem_alloc(host_mem.nbytes)
                    else:   # fixed shape
                        # rich.print(f'>>> shape: {shape} | dtype: {dtype}')
                        host_mem = np.empty(shape, dtype)
                        device_mem = cuda.mem_alloc(host_mem.nbytes)
                    
                    self.bindings.append(int(device_mem))  # save input device mem
                    self.inputs.append({'host': host_mem, 'device': device_mem})  # save inputs mem

                # outputs
                else:
                    if self.dynamic:
                        host_mem = np.empty(self.dynamic_max_shape, dtype)   # max dynamic shape
                        device_mem = cuda.mem_alloc(host_mem.nbytes)   
                    else:
                        host_mem = np.empty(shape, dtype)
                        device_mem = cuda.mem_alloc(host_mem.nbytes)
                    self.bindings.append(int(device_mem))  # save output device mem
                    self.outputs.append({'host': host_mem, 'device': device_mem})  # save outputs mem


            # # test
            # for i in range(self.model.num_bindings):

            #     dtype = trt.nptype(self.model.get_binding_dtype(i))
            #     # size = trt.volume(self.model.get_binding_shape(i))
            #     name = self.model.get_binding_name(i)
            #     shape = self.model.get_binding_shape(i)
            #     # info
            #     rich.print(f"test> {i} | {name} | {shape} | {dtype} | context shape: {self.context.get_binding_shape(i)}")
            #     rich.print(f"---> {self.model.get_binding_dtype(i)}")

            rich.print(f"dynamic: {self.dynamic} | fp16: {self.fp16} | fp32: {self.fp32} | int8: {self.int8}")
            # sys.exit()

            self.classes_names =  None


        else:
            LOGGER.error(f"Weights not load successfullly!")
            exit()

        # check classes_names again
        if self.classes_names is None:
            self.classes_names = {i: f'class{i}' for i in range(100)}

        # check input_size shape again -> (x, x)
        if isinstance(self.input_size, int):
            self.input_size = (self.input_size, self.input_size)

        # warmup
        if do_warmup:
            self._warmup()



    def _to_device(self, device: DeviceInfo):
        # device change

        self.is_device_shifted = True  # mark it as shifted
        
        if self.onnx:
            self.device = device  # update self.device
            
            # update onnx session provider
            if device.type == 'cpu':
                self.session.set_providers(
                    providers = ['CPUExecutionProvider']
                )
                self.io_binding = self.session.io_binding()  # re-binding
            elif device.type == 'cuda':
                _cuda_provider = ( 
                    'CUDAExecutionProvider', {
                    'device_id': device.id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 5 * GB,  # TODO
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                })

                self.session.set_providers(
                    providers = [_cuda_provider, 'CPUExecutionProvider']
                )
                self.io_binding = self.session.io_binding()     # re-binding
    
        else:  # other backends
            pass



    def pre_process(self, x, size):
        # batch images pre-process

        # checking
        assert isinstance(x, list), "--> Input images must be a list!"  # make sure input images is list
        assert len(x) > 0, "--> Input images is empty!"  # make sure has input images

        # preprocess
        im_list, im0_list = [], []
        for image in x:
            im, im0 = self._pre_process_single_image(image, size)   # pre-process single image
            im_list.append(im)
            im0_list.append(im0)

        #  batch_images, original_image_list --->  cpu data
        return np.concatenate(im_list, 0), im0_list  # (batch, 3, size, size ), original image list(np.ndarray)


    def _pre_process_single_image(self, x, size):
        if isinstance(x, (str, np.ndarray)):

            # read image if it is not ndarray
            if isinstance(x, str):   
                x = cv2.imread(x)
            assert isinstance(x, np.ndarray), f'Image type should be np.ndarray'

            # pre-process for all types of models
            if self.is_detector:
                return self._pre_process_detector(x, size)
            elif self.is_classifier:
                return self._pre_process_classifier(x, size)
            elif self.is_other:  # others
                return self.pre_process_other(x, size)

        else:
            LOGGER.error(f'Type of element in Input image list is wrong! ---> {x}')
            exit()



    def _pre_process_detector(self, x, size):
        # default pre-process for detector, support override

        y, _, (pad_w, pad_h) = letterbox(x, new_shape=size, auto=False) 
        y = np.ascontiguousarray(y.transpose((2, 0, 1))[::-1]).astype(dtype=np.float32) / 255.0 # HWC to CHW -> BGR to RGB -> contiguous 
        y = y[None] if len(y.shape) == 3 else y
        return y, x


    def _pre_process_classifier(self, x, size):
        # default pre-process for classifier, support override

        y = CenterCrop(size)(x)  # crop
        y = np.ascontiguousarray(y.transpose((2, 0, 1))[::-1]).astype(dtype=np.float32) / 255.0 # HWC to CHW -> BGR to RGB -> contiguous 
        y = Normalize()(y)
        y = y[None] if len(y.shape) == 3 else y     # add batch dim
        return y, x


    def pre_process_other(self, x, size):
        # interface for other type, support override
        ...


    # TODO 
    # def auto_infer(self, ims, im0s):
    def auto_infer(self):
        '''
        ----------------------------------------------------------------------
            detector output format:  
        ----------------------------------------------------------------------
        [
            array([[x1, y1, x2, y2, conf, class_id],
                   [x1, y1, x2, y2, conf, class_id]], dtype=float32),   # image_1 
            array([], shape=(0, 6), dtype=float32)  # image_2 
        ]
        ----------------------------------------------------------------------
            classifier output format:  
        ----------------------------------------------------------------------
        [
            array([[class_id, class_name, conf],  # image_1 
                   [class_id, class_name, conf],  # image_2 
                   [class_id, class_name, conf]], dtype=float32)  # image_3 
        ]
        '''

        if self.onnx:

            # create input & output ort value
            if self.ort_x is None or self.is_device_shifted or self.ims.shape != self.ort_x.shape():
                self.is_device_shifted = False  # mark it false

                # TODO: destroy original memory
                # pre-allocate cuda input & output mem
                self.ort_x = onnxruntime.OrtValue.ortvalue_from_numpy(
                    numpy_obj=self.ims, 
                    device_type=self.device.type, 
                    device_id=self.device.id
                )
                self.ort_y = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
                    shape=self.ort_x.shape()[:1] + self.output_shape[1:],
                    element_type=self.ims.dtype, 
                    device_type=self.device.type, 
                    device_id=self.device.id
                )    
            else:  # update input ort value
                self.ort_x.update_inplace(self.ims)   


            # input & output binding
            self.io_binding.bind_ortvalue_input(self.input_names[0], self.ort_x)
            self.io_binding.bind_ortvalue_output(self.output_names[0], self.ort_y)
    
            
            # run & post-process
            if self.is_detector:    # det
                # with TIMER('run_with_iobinding'):
                self.session.run_with_iobinding(self.io_binding)  # infer, an OrtValue which has data allocated by ONNX Runtime on CUDA
                    
                # LOGGER.info(f"out onnx ---> {self.ort_y.numpy().shape}")
                # with TIMER(f'nms time'):

                y = batched_nms(self.ort_y.numpy(), self.conf_threshold, self.nms_threshold)
                y = scale_boxes_batch(y, self.ims, self.im0s)  # de-scale
                return np.asarray(y, dtype=object)    # numpy.ndarray  

            elif self.is_classifier:    # cls
                self.session.run_with_iobinding(self.io_binding)
                # y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: ims})[0]  # infer   (batch_size, num_class)
                
                res = []
                for i, x in enumerate(self.ort_y.numpy()):
                    prob = Softmax(dim=0)(x)
                    if max(prob) < self.conf_threshold:
                        res.append([])
                    else:
                        pred_cls = np.argmax(prob)
                        res.append([pred_cls, self.classes_names[pred_cls], prob[pred_cls]])
                return np.asarray(res)  # numpy.ndarray


            elif self.is_other:  # other type model
                # return self._infer_by_others(x)
                pass


        elif self.pt:  # Not Now
            pass


        elif self.engine:   # TRT
            # > self.ims is input!!!  default has one input

            # input data dtype modify
            if self.fp16:
                self.xs = np.ascontiguousarray(self.ims.astype(np.float16))
            elif self.int8:
                self.xs = np.ascontiguousarray(self.ims.astype(np.int8))
            else:
                self.xs = np.ascontiguousarray(self.ims)

            # for detector
            if self.is_detector:    # det
                
                # self.ctx.push()
                
                with TIMER(f"data-prepare"):

                    # inputs 
                    for i in range(self.num_inputs):

                        # dynamic -> re-malloc
                        if self.dynamic and self.ims.shape != self.dynamic_max_shape:
                            self.context.set_binding_shape(i, self.ims.shape)   # set context shape
                            dtype = trt.nptype(self.model.get_binding_dtype(i))
                            host_mem = self.xs
                            device_mem = cuda.mem_alloc(host_mem.nbytes)  # re-malloc device mem
                            self.bindings[i] = int(device_mem)  # update input device mem
                            self.inputs[i] = {'host': host_mem, 'device': device_mem}  # update
                        # fixed shape -> assigne input image
                        else:
                            self.inputs[i]['host'] = self.xs   # self.imgs is always the 1st input


                    # outputs mem -> dynamic -> re-malloc
                    if self.dynamic:
                        for i in range(self.num_outputs):
                            dtype = trt.nptype(self.model.get_binding_dtype(self.num_inputs + i))
                            shape = self.context.get_binding_shape(self.num_inputs + i)
                            host_mem = np.empty(shape, dtype)   # re-malloc host mem
                            device_mem = cuda.mem_alloc(host_mem.nbytes)  # re-malloc device mem
                            self.bindings[self.num_inputs + i] = int(device_mem)  # save output device mem
                            self.outputs[i] = {'host': host_mem, 'device': device_mem}  # save outputs mem


                    # rich.print(f"inputs host emm shape: {self.inputs[0]['host'].shape}")
                    # rich.print(f"inputs host emm shape: {self.inputs[0]['host'].dtype}")
                    # rich.print(f"inputs host emm shape: {self.inputs[0]['device']}")

                    # rich.print(f"outputs host emm shape: {self.outputs[0]['host'].shape}")
                    # rich.print(f"outputs host emm shape: {self.outputs[0]['host'].dtype}")
                    # rich.print(f"outputs host emm shape: {self.outputs[0]['device']}")

                    # sys.exit()


                with TIMER(f"memcopy host->device"):

                    # host -> device
                    for mem in self.inputs:
                        cuda.memcpy_htod_async(mem['device'], mem['host'], self.stream)


                with TIMER(f"trt infer & synchronize"):
                    # infer
                    self.context.execute_async_v2(
                        bindings=self.bindings,
                        stream_handle=self.stream.handle
                    )
                    self.stream.synchronize()

                with TIMER(f"memcopy device->host"):
                    for mem in self.outputs:
                        cuda.memcpy_dtoh_async(mem['host'], mem['device'], self.stream)
                    y = [mem['host'] for mem in self.outputs] if self.num_outputs > 0 else self.outputs[0]['host']



                with TIMER(f"post-process"):

                    y = batched_nms(y, self.conf_threshold, self.nms_threshold)
                    y = scale_boxes_batch(y, self.ims, self.im0s)  # de-scale
                    
                # rich.print(y)

                return np.asarray(y, dtype=object)    # numpy.ndarray 








                '''

                # poset process
                assert len(data) == 4
                num_dets, bboxes, scores, labels = (i[0] for i in data)
                nums = num_dets.item()
                bboxes = bboxes[:nums]
                scores = scores[:nums]
                labels = labels[:nums]


                # LOGGER.info(f"---> post process bboxes: {bboxes}")
                # LOGGER.info(f"---> post process scores: {scores}")
                # LOGGER.info(f"---> post process labels: {labels}")

                return bboxes, scores, labels

                '''



            elif self.is_classifier:    # cls
                pass

            elif self.is_other:
                return self._infer_by_others(x)


        else:
            pass


    def _infer_by_others(self, x):
        # other type models inference
        pass


    # def post_process(self, x):
    #   code here
    #     return x


    def __call__(self, x, do_post_process=True):
        # x -> dataloader

        # with INSPECTOR('pre_process'):
        self.ims, self.im0s = self.pre_process(x, size=self.input_size)  # pre process for all type model

        with INSPECTOR('auto_infer'):
            y = self.auto_infer()  # multi infer

        return y
        # return self.post_process(y) if do_post_process else y  # post process   no need


    def _get_custom_methods(self):
        # get custom methods

        custom_methods_list = []
        for x in dir(self):
            if not x.startswith('__') and not x.endswith('__') and callable(eval("self." + x)):
                custom_methods_list.append(x)
        return custom_methods_list


    def __repr__(self):
        return self.__class__.__name__


    def __set_name__(self, owner, func_name):
        pass        


    def _warmup(self, input_shape=None, times=1, batch_size=3):
        # warmup

        if not input_shape:
            input_shape=tuple(self.input_size) + (3,)
        ims = [np.random.randint(low=0, high=255, size=input_shape)] * batch_size  # , dtype=np.float32)
        for _ in range(times):
            _ = self(ims)
        # LOGGER.info(f'> {self.__class__.__name__} class done warmup!')



 




