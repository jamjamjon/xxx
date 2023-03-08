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

            # Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            # logger = trt.Logger(trt.Logger.INFO)
            # with open(self.model_weights, 'rb') as f, trt.Runtime(logger) as runtime:
            #     self.model = runtime.deserialize_cuda_engine(f.read())
            # self.context = self.model.create_execution_context()
            # self.bindings = OrderedDict()
            # self.output_names = []
            # self.fp16 = False  # default updated below
            # self.dynamic = False
            # for i in range(self.model.num_bindings):
            #     name = self.model.get_binding_name(i)
            #     dtype = trt.nptype(self.model.get_binding_dtype(i))

            #     if self.model.binding_is_input(i):
            #         if -1 in tuple(self.model.get_binding_shape(i)):  # dynamic
            #             self.dynamic = True
            #             self.context.set_binding_shape(i, tuple(self.model.get_profile_shape(0, i)[2]))
            #         if dtype == np.float16:
            #             self.fp16 = True
            #     else:  # output
            #         self.output_names.append(name)

            #     shape = tuple(self.context.get_binding_shape(i))



            #     self.device_torch = torch.device('cuda:' + str(self.device.id))
            #     print(f'device : {self.device_torch} | type({type(self.device_torch)})')


            #     im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device_torch)
            #     self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
                
            # self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
            # # batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size

            # rich.print(f"dynamic: {self.dynamic} | fp16: {self.fp16}")


            # # check io info 
            # for i in range(self.model.num_bindings):
            #     print(
            #         f"name: {self.model.get_binding_name(i)} | shape: {self.model.get_binding_shape(i)} | dtype: {self.model.get_binding_dtype(i)} "
            #         f" | shape: {self.context.get_binding_shape(i)} "
            #     )
                

            self.stream = cuda.Stream(0)

            # initial engine
            logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(logger, namespace='')
            with trt.Runtime(logger) as runtime:
                self.model_weights = Path(self.model_weights) if isinstance(self.model_weights, str) else self.model_weights

                model = runtime.deserialize_cuda_engine(self.model_weights.read_bytes())
            context = model.create_execution_context()
            names = [model.get_binding_name(i) for i in range(model.num_bindings)]
            self.num_bindings = model.num_bindings
            self.bindings: List[int] = [0] * self.num_bindings
            num_inputs, num_outputs = 0, 0

            for i in range(model.num_bindings):
                if model.binding_is_input(i):
                    num_inputs += 1
                else:
                    num_outputs += 1

            self.num_inputs = num_inputs
            self.num_outputs = num_outputs
            self.model = model
            self.context = context
            self.input_names = names[:num_inputs]
            self.output_names = names[num_inputs:]


            # init binding
            dynamic = False
            Tensor = namedtuple('Tensor', ('name', 'dtype', 'shape', 'cpu', 'ptr'))
            inp_info = []
            out_info = []
            out_ptrs = []
            
            
            # new 
            for i in range(self.model.num_bindings):
            
                dtype = trt.nptype(self.model.get_binding_dtype(i))
                shape = tuple(self.model.get_binding_shape(i))
                name = self.model.get_binding_name(i) 
                rich.print(f'shape: {shape}')

                if self.model.binding_is_input(i):
                    
                    if -1 in shape:
                        dynamic = True
                    if not dynamic:
                        cpu = np.empty(shape, dtype)
                        gpu = cuda.mem_alloc(cpu.nbytes)
                        cuda.memcpy_htod_async(gpu, cpu, self.stream)
                    else:

                        # [(1, 3, 640, 640), (2, 3, 640, 640), (8, 3, 640, 640)]
                        # rich.print('---> profile shape', model.get_profile_shape(0, i)) 
                        # self.context.set_binding_shape(i, tuple(self.model.get_profile_shape(0, i)[2]))  # max batch size
                        # shape = tuple(self.context.get_binding_shape(i))  # get current shape
                        # cpu = np.empty(shape, dtype)
                        # gpu = cuda.mem_alloc(cpu.nbytes)
                        # cuda.memcpy_htod_async(gpu, cpu, self.stream)

                        cpu, gpu = np.empty(0), 0

                    inp_info.append(Tensor(name, dtype, shape, cpu, gpu))
                else:
                    if not dynamic:
                        cpu = np.empty(shape, dtype=dtype)
                        gpu = cuda.mem_alloc(cpu.nbytes)
                        cuda.memcpy_htod_async(gpu, cpu, self.stream)
                        out_ptrs.append(gpu)
                    else:
                        cpu, gpu = np.empty(0), 0
                    out_info.append(Tensor(name, dtype, shape, cpu, gpu))


            self.is_dynamic = dynamic
            self.inp_info = inp_info
            self.out_info = out_info
            self.out_ptrs = out_ptrs

            rich.print(f"dynamic: {self.is_dynamic}")


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
            if self.is_detector:    # det


                # if self.dynamic and self.ims.shape != self.bindings['images'].shape:


                #     i = self.model.get_binding_index('images')
                #     self.context.set_binding_shape(i, self.ims.shape)  # reshape if dynamic
                #     self.bindings['images'] = self.bindings['images']._replace(shape=self.ims.shape)

                #     for name in self.output_names:
                #         i = self.model.get_binding_index(name)
                #         self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))


                # s = self.bindings['images'].shape


                # # check io info 
                # for i in range(self.model.num_bindings):
                #     print(
                #         f"name: {self.model.get_binding_name(i)} | shape: {self.model.get_binding_shape(i)} | dtype: {self.model.get_binding_dtype(i)} "
                #         f" | shape: {self.context.get_binding_shape(i)} "
                #     )
                

                # assert self.ims.shape == s, f"input size {self.ims.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"


                # ims = torch.from_numpy(self.ims).to(self.device_torch)  # TODO

                # self.binding_addrs['images'] = int(ims.data_ptr())



                # self.context.execute_v2(list(self.binding_addrs.values()))
                # y = [self.bindings[x].data for x in sorted(self.output_names)]
                
                # rich.print(f"y----> {len(y)}")
                # rich.print(f"y----> {y[0].shape}")
                # rich.print(f"y----> {type(y[0])}")


                # yy = batched_nms(y[0].cpu().numpy(), self.conf_threshold, self.nms_threshold)
                # LOGGER.info(f"engine yyyyyyyyy ---> {yy}")


                # yy = scale_boxes_batch(yy, self.ims, self.im0s)  # de-scale


                # LOGGER.info(f"engine yyyyyyyyy ---> {yy}")

                # sys.exit()


                # assert len(self.ims) == self.num_inputs
                # rich.print(f'self.num_inputs: {self.num_inputs}')
                # rich.print(f'len(self.ims): {len(self.ims)} | shape: {self.ims.shape}')
                

                # contiguous_inputs: List[ndarray] = [
                #     np.ascontiguousarray(i) for i in self.ims
                # ]

                contiguous_inputs = [np.ascontiguousarray(self.ims)]
                

                # rich.print(f"contiguous_inputs222222222: {contiguous_inputs.shape}")
                # rich.print(f"contiguous_inputs222222222: {len(contiguous_inputs)}")
                # rich.print(f"contiguous_inputs222222222: {len(contiguous_inputs[0])}")
                # rich.print(f"contiguous_inputs22222222222: {contiguous_inputs[0].shape}")
                # sys.exit()

                # rich.print(f"num inputs:: {self.num_inputs}")

                for i in range(self.num_inputs):
                    # if self.is_dynamic:
                    #     self.context.set_binding_shape(
                    #         i, 
                    #         # tuple(contiguous_inputs[i].shape)  
                    #         tuple(self.model.get_profile_shape(0, i)[2])   # max batch size
                    #     )

                    if self.is_dynamic and self.ims.shape != self.model.get_binding_shape(i):
                        self.context.set_binding_shape(
                            i, 
                            tuple(contiguous_inputs[i].shape)   # tuple(self.model.get_profile_shape(0, i)[2])   # max batch size
                            
                        )

                        print(
                            f"name: {self.model.get_binding_name(i)} | shape: {self.model.get_binding_shape(i)} | dtype: {self.model.get_binding_dtype(i)} "
                            f" | shape: {self.context.get_binding_shape(i)} "
                        )


                        rich.print(f'{i} ::: ', self.inp_info[i].ptr) 
                        rich.print(f'{i} contiguous_inputs[i].nbytes::: ', contiguous_inputs[i].nbytes) 
                        _gpu = int(contiguous_inputs[i].nbytes)
                        self.inp_info[i].ptr = cuda.mem_alloc(_gpu)


                        sys.exit()


                    cuda.memcpy_htod_async(
                        self.inp_info[i].ptr, 
                        contiguous_inputs[i],
                        self.stream
                    )
                    self.bindings[i] = int(self.inp_info[i].ptr)


                # check io info 
                for i in range(self.model.num_bindings):
                    print(
                        f"name: {self.model.get_binding_name(i)} | shape: {self.model.get_binding_shape(i)} | dtype: {self.model.get_binding_dtype(i)} "
                        f" | shape: {self.context.get_binding_shape(i)} "
                        )

                sys.exit()



                output_gpu_ptrs: List[int] = []
                outputs: List[ndarray] = []


                for i in range(self.num_outputs):
                    j = i + self.num_inputs
                    if self.is_dynamic:
                        shape = tuple(self.context.get_binding_shape(j))
                        dtype = self.out_info[i].dtype
                        cpu = np.empty(shape, dtype=dtype)
                        gpu = cuda.mem_alloc(contiguous_inputs[i].nbytes)
                        cuda.memcpy_htod_async(gpu, cpu, self.stream)
                    else:
                        cpu = self.out_info[i].cpu
                        gpu = self.out_info[i].gpu
                    outputs.append(cpu)
                    output_gpu_ptrs.append(gpu)
                    self.bindings[j] = int(gpu)


                # LOGGER.info(f"engine before infer")


                # check io info 
                for i in range(self.model.num_bindings):
                    print(
                        f"name: {self.model.get_binding_name(i)} | shape: {self.model.get_binding_shape(i)} | dtype: {self.model.get_binding_dtype(i)} "
                        f" | shape: {self.context.get_binding_shape(i)} "
                        )



                self.context.execute_async_v2(self.bindings, self.stream.handle)
                self.stream.synchronize()

                # LOGGER.info(f"engine after infer")


                for i, o in enumerate(output_gpu_ptrs):
                    cuda.memcpy_dtoh_async(outputs[i], o, self.stream)


                # LOGGER.info(f"engine output ---> {len(outputs)}")


                # return tuple(outputs) if len(outputs) > 1 else outputs[0]
                data = tuple(outputs) if len(outputs) > 1 else outputs[0]

                # LOGGER.info(f"engine data ---> {type(data)}")
                # LOGGER.info(f"engine data ---> {len(data)}")
                # LOGGER.info(f"engine data ---> {data.shape}")

                
                y = batched_nms(data, self.conf_threshold, self.nms_threshold)
                y = scale_boxes_batch(y, self.ims, self.im0s)  # de-scale
                
                
                # LOGGER.info(f"engine yyyyyyyyy ---> {y}")
                
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

        # with INSPECTOR('auto_infer'):
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



 




