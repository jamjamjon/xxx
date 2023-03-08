import numpy as np
import numpy
import cv2
from omegaconf import DictConfig, OmegaConf
import contextlib
import rich
from pathlib import Path
from flask import Flask, request, jsonify
import rich
from rich.console import Console
from rich.live import Live
import psutil
from datetime import datetime
import time
import sys
import inspect
import os
from itertools import chain
# from numba import jit, njit
import logging
import pynvml
import onnxruntime
import weakref
from dataclasses import dataclass
from typing import Union, Any
import re
# from pympler import asizeof as pysizeof
# sizeof = pysizeof.asizeof



# ---------------------------------------------------------------------------
# global variables
WEIGHTS_TYPE = ('.onnx', '.pt', '.engine')
MODEL_TYPE = ('detector', 'classifier', 'other')
IMG_FORMAT = ('.bmp', '.jpg', '.jpeg', '.png')
VIDEO_FORMAT = ('.mp4', '.flv', '.avi', '.mov')
STREAM_FORMAT = ('rtsp://', 'rtmp://', 'http://', 'https://')
FLASK_APP = Flask(__name__)
CONSOLE = Console()
GB, MB, KB = 1 << 30, 1 << 20, 1 << 10
LOGGER = logging.getLogger(__name__)   # hydra logging


# ---------------------------------------------------------------------

@dataclass
class HostDeviceMemory:
    # host device memory
    host: Any
    device: Any




class Visualizer:
    # use cv2

    def __init__(self, line_width=None, color=(128, 255, 128), txt_color=(255, 255, 255)):
        self.lw = line_width
        self.color = color
        self.txt_color = txt_color


    def draw(self, im0, box=None, label=None, conf=None):

        self.im = im0   # im0.copy()
        self.lw = self.lw if self.lw else max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

        if box is not None:     # for detection
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

            # assign color for different classes
            if label is not None:
                assert isinstance(label, (int, str, float))
                self.color = Colors()(sum([ord(x) for x in label])) if isinstance(label, str) else Colors()(label)
            
            cv2.rectangle(self.im, p1, p2, self.color, thickness=self.lw, lineType=cv2.LINE_AA)  # draw bbox

            # draw text
            if label is not None:
                label = str(label) + ' ' + str(conf) if conf else ''  # label = cls + conf
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, self.color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    self.im,
                    label, 
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    self.lw / 3,
                    self.txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA
                )

        elif box is None and label is not None:     # for classification
            h, w = self.im.shape[:-1]
            label = str(label) + ' ' + str(conf) if conf else ''  # label = cls + conf
            tf = max(self.lw - 1, 1)  # font thickness
            cv2.putText(
                self.im, 
                label,
                (w // 10, h // 5), 
                0, 
                self.lw / 3, 
                self.color, 
                thickness=tf, 
                lineType=cv2.LINE_AA
            )


        elif not all((box, label, conf)):
            LOGGER.warning('No elements to visualize!')
            return 


        # TODO: cv2.imahow()



class Colors:
    '''
        colors palette
        hex 颜色对照表    https://www.cnblogs.com/summary-2017/p/7504126.html
        RGB的数值 = 16 * HEX的第一位 + HEX的第二位
        RGB: 92, 184, 232 
        92 / 16 = 5余12 -> 5C
        184 / 16 = 11余8 -> B8
        232 / 16 = 14余8 -> E8
        HEX = 5CB8E8
    '''

    def __init__(self, shuffle=False):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('33FF00', '9933FF', 'CC0000', 'FFCC00', '99FFFF', '3300FF', 'FF3333', # new add
               'FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', 
               '1A9334', '00D4BB', '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', 
               '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        
        # shuffle color 
        if shuffle:
            hex_list = list(hex)
            random.shuffle(hex_list)
            hex = tuple(hex_list)

        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)
        # self.b = random   # also for shuffle color 


    def __call__(self, i, bgr=False):        
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod  
    def hex2rgb(h):  # int('CC', base=16) 将16进制的CC转成10进制 
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))




@dataclass
class DeviceInfo:
    # gpu device 
    id: int = 0
    type: str = 'cpu'



@dataclass
class InstanceInfo:
    # Instance Attributes Info

    device: DeviceInfo
    weakref: weakref.ReferenceType  # weakref.ProxyType    # 
    # hash_id: int
    uuid: str   # uuid
    ptr: str    # mem address
    cur_class_name: str  # current class name
    base_class_names: list     # base class names
    file_name: str   # instance init file name
    lineno: int     # instance init line number
    date: str   # instance init time
    component_name: str = 'default' # TODO: keep or delete
    # device: Union[str, int] = 'cpu'  # TODO:






def get_device(device: Union[int, str] = 'cpu') -> DeviceInfo:
    # support device='0', device=0, device='cuda:0', device='cpu', ..
    # not support multi device: device='0,1,2'

    # test code
    # print(get_device(1))
    # print(get_device('cuda: 999'))
    # print(get_device(' cuda:0 '))
    # print(get_device('1'))
    # print(get_device('cpu'))



    device = str(device).strip().lower().replace('cuda:', '').replace(' ', '')  # to string, "cuda:1" -> '1'
    is_cpu = device == 'cpu'
    is_mps = device == 'mps'

    if is_cpu:
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False 
        return DeviceInfo(id=0, type='cpu')
    elif device:
        try:
            pynvml.nvmlInit()
            num_gpu = pynvml.nvmlDeviceGetCount()
            # os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
            assert int(device) < num_gpu, f"Please check device_id, greater than {num_gpu}"
            return DeviceInfo(id=int(device), type='cuda')
        except Exception:
            LOGGER.error(f"This machine has no GPU device, `device: {device}` will automatically be set `device.type=cpu, device.id=0`!")
            return DeviceInfo(id=0, type='cpu')
    elif is_mps:
        return DeviceInfo(id=0, type='mps')



def build_resolver():
    # build omegaconf resolver
    OmegaConf.register_new_resolver("concat", lambda x, y: x + y)
    # add others here




def gpu_info(display=False):

    try:
        pynvml.nvmlInit()  # init
    except Exception:
        LOGGER.error(f"This machine has no GPU device!")
        return   

    # create table
    table = rich.table.Table(title="\n[bold cyan]GPU INFO", 
                            box=rich.box.ASCII2, 
                            show_lines=False, 
                            caption=f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}\n",  # Time
                            caption_justify='center',
                        )

    # add table column
    table.add_column("ID", justify="center", style="cyan", no_wrap=True)
    table.add_column("NAME", justify="center", style="cyan", no_wrap=True)
    table.add_column("USED", justify="center", style="cyan", no_wrap=True)
    table.add_column("TOTAL", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"USAGE", justify="center", style="cyan", no_wrap=True)

    # init
    pynvml.nvmlInit()  
    for index in range(pynvml.nvmlDeviceGetCount()):   # num_gpu
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)  # handle
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)  # gpu mem

        # add table row
        table.add_row(
            f"{index}", 
            f"{str(pynvml.nvmlDeviceGetName(handle), encoding='utf-8')}",  
            f"{(mem.used / MB):.3f} MB",  
            f"{(mem.total / MB):.3f} MB",  
            f"{(mem.used / MB) / (mem.total / MB) * 100:.4f} %",
            end_section=True
        )

    # display
    if display:
        CONSOLE.print(table)

    pynvml.nvmlShutdown()  # close
    return table



def resource_info(refresh_time=0.5, display=False):

    # cpu info
    cpu_count = psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)  # logical, virtual
    cpu_usage = psutil.cpu_percent(percpu=True, interval=refresh_time)
    cpu_usage_avg = psutil.cpu_percent(percpu=False, interval=refresh_time)
    cpu_load_average = psutil.getloadavg()  # sum(cpu_usage) / cpu_count[1]


    # mem info
    mem = psutil.virtual_memory()
    mem_swap = psutil.swap_memory()

    # create table
    table = rich.table.Table(title="\n[bold cyan]CPU & MEM INFO", 
                            box=rich.box.ASCII2, 
                            show_lines=False, 
                            caption=f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} (refresh time: {refresh_time})\n",  # Time
                            caption_justify='center',
                        )

    # add table column
    table.add_column(f"CPU\nUSAGE", justify="center", style="cyan", no_wrap=True)
    # table.add_column(f"CPU\nUSAGE_PER_CORE", justify="center", style="cyan", no_wrap=True)
    table.add_column("CPU\nCOUNT", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"CPU\nLOAD_AVERAGE", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"MEM\nUSAGE", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"MEM\nTOTAL(GB)", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"MEM\nUSED(GB)", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"MEM\nAVAILABLE(GB)", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"MEM_SWAP\nUSAGE(GB)", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"MEM_SWAP\nTOTAL(GB)", justify="center", style="cyan", no_wrap=True)

    # add table row
    table.add_row(
        f"{cpu_usage_avg}%", 
        # f"{cpu_usage}", 
        f"{cpu_count}",
        f"{cpu_load_average}%", 
        f"{mem.percent}%", 
        f"{(mem.total / (1 << 30)):.3f}",   # 2^30, kb=2^10
        f"{(mem.used / (1 << 30)):.3f}", 
        f"{(mem.available / (1 << 30)):.3f}", 
        f"{mem_swap.percent}%", 
        f"{(mem_swap.total / (1 << 30)):.4}", 
        end_section=True
    )

    # display
    if display is True:
        CONSOLE.print(table)


    return table



class INSPECTOR(contextlib.ContextDecorator):
    # timer decorator

    def __init__(self, prefix='Timer', repeat=0):
        self.prefix = prefix
        self.repeat = repeat


    def __enter__(self):
        self.t0 = time.time()
        return self


    def __exit__(self, type, value, traceback):
        # rich.print(inspect.stack())
        self.duration = time.time() - self.t0
        LOGGER.info(f"[INSPECTOR] {self.prefix} consume: {(time.time() - self.t0) * 1e3:.2f} ms.")

        # TODO: repeat


    def __call__(self, func):
        def wrapper(*args, **kwargs):
            for i in range(self.repeat):  
                t0 = time.time()
                ret = func(*args, **kwargs)
                LOGGER.info(f"[INSPECTOR][{i + 1}-th] {self.prefix} consume: {(time.time() - t0) * 1e3:.2f} ms.")
            return ret
        return wrapper



class TIMER(contextlib.ContextDecorator):
    # timer decorator

    def __init__(self, prefix='Time'):
        self.prefix = prefix


    def __enter__(self):
        self.t0 = time.time()
        return self


    def __exit__(self, type, value, traceback):
        # rich.print(inspect.stack())
        self.duration = time.time() - self.t0
        LOGGER.info(f"[TIMER] {self.prefix} consume: {(time.time() - self.t0) * 1e3:.2f} ms.")


    def __call__(self, func):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            ret = func(*args, **kwargs)
            LOGGER.info(f"[TIMER] {self.prefix} consume: {(time.time() - t0) * 1e3:.2f} ms.")

            return ret
        return wrapper





# class INSPECTOR(contextlib.ContextDecorator):

#     def __init__(self, prefix='Time', cpu=False, mem=False, gpu=False):
#         self.prefix = prefix
#         self.cpu, self.mem, self.gpu = cpu, mem, gpu


#     def __enter__(self):
#         self.t0 = time.time()

#         if self.cpu:
#             self.cpu_usage_avg0 = psutil.cpu_percent(percpu=False, interval=None)

#         if self.mem:
#             self.mem0 = psutil.virtual_memory().used

#         if self.gpu:
#             pass

#         return self


#     def __exit__(self, type, value, traceback):
#         self.duration = time.time() - self.t0  
#         CONSOLE.log(f"{self.prefix} | Time consume: {(time.time() - self.t0) * 1e3:.2f} ms.")

#         if self.cpu:
#             CONSOLE.log(f"{self.prefix} | CPU cost: {(psutil.cpu_percent(percpu=False, interval=None) - self.cpu_usage_avg0)} %.")
#             # CONSOLE.log(f"{self.prefix} | start: {self.cpu_usage_avg0} | end: {(psutil.cpu_percent(percpu=False, interval=None))} | CPU cost: {(psutil.cpu_percent(percpu=False, interval=None) - self.cpu_usage_avg0)} %.")

#         if self.mem:
#             CONSOLE.log(f"{self.prefix} | Memory cost: {(psutil.virtual_memory().used - self.mem0) / GB:.3f} %.")
#             # CONSOLE.log(f"{self.prefix} | start: {self.mem0 / GB} | end: {psutil.virtual_memory().used / GB} | Memory cost: {(psutil.virtual_memory().used - self.mem0) / GB:.3f} %.")

#         if self.gpu:
#             pass



#     def __call__(self, func):
#         def wrapper(*args, **kwargs):
#             t0 = time.time()
#             ret = func(*args, **kwargs)
#             CONSOLE.log(f"{self.prefix} consume: {(time.time() - t0) * 1e3:.2f} ms.")
#             return ret
#         return wrapper




@TIMER(prefix="get_projects_configs")
def get_projects_configs(configs=None, has_common=False):
    # clean up all projects configs, merge common settings
    # k -> model_name(project dir name)
    # v -> model_class, model_instance_name, model_config


    x = OmegaConf.create()
    for k, v in configs.items():

        if has_common:
            if k == 'common':
                continue

            # load common first
            OmegaConf.update(x, k, {'model_class': v['model_class'],     # build class  !! need!!!!
                                    'model_instance_name': k + '_model',   #  TODO  : more instance names  [, , ]
                                    'model_config': configs.common})

            # update `model_config`
            OmegaConf.update(x[k], 'model_config', v)

        else:
            OmegaConf.update(x, k, {'model_class': v['model_class'], 
                                    'model_instance_name': k + '_model',
                                    'model_config': v})

        # add ai rule
        OmegaConf.update(x[k], 'model_config', {'component_name': k})


    return OmegaConf.to_container(x, resolve=True)  # to primitive container



def parse_model_config(cfg): 
    # backup
    # parse cfg before setup model

    if not cfg.get('modules', None):   # single module: detector or classifier only
        args, common_args = [cfg], None
        return args
    else:   # multi sequencial modules: e.g. detecting then classify
        args, common_args = None, {}
        for k, v in cfg.items():
            if k == 'modules':
                args = v
            else:
                common_args.update({k:v})

        config = []
        for arg in args:
            for k, v in arg.items():
                v = OmegaConf.merge({'model_name': k}, v)   # add `model_name` key 
                config.append(OmegaConf.merge(common_args, v))
        return config






class CenterCrop:
    def __init__(self, size=224):
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top:top + m, left:left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, x): 
        if not isinstance(self.mean, np.ndarray):
            self.mean = np.array(self.mean, dtype=np.float32)
        if not isinstance(self.std, np.ndarray):
            self.std = np.array(self.std, dtype=np.float32)
        if self.mean.ndim == 1:
            self.mean = np.reshape(self.mean, (-1, 1, 1))
        if self.std.ndim == 1:
            self.std = np.reshape(self.std, (-1, 1, 1))
        _max = np.max(abs(x))
        _div = np.divide(x, _max)  # i.e. _div = data / _max
        _sub = np.subtract(_div, self.mean)  # i.e. arrays = _div - mean
        arrays = np.divide(_sub, self.std)  # i.e. arrays = (_div - mean) / std
        return arrays


class Softmax:
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def __call__(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=self.dim)



def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y



def scale_boxes_batch(boxes, ims_shape, im0s_shape):
    # Batch Rescale boxes (xyxy) to original image size

    for i in range(len(boxes)):
        if len(boxes) > 0:
            boxes[i][:, :4] = scale_boxes(ims_shape[i].shape[1:], boxes[i][:, :4], im0s_shape[i].shape[:-1]) # .round()
    return boxes



def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape

    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain

    # clip boxes
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img0_shape[0])  # y1, y2
    return boxes



# @jit
def batched_nms(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, nm=0):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # with TIMER('prepare'):
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [np.zeros((0, 6 + nm), dtype=np.float32)] * bs

    for xi, x in enumerate(prediction):  # image index, image inference
        # with TIMER(f'for - {xi}'):
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:mi].max(1, keepdims=True), x[:, 5:mi].argmax(1, keepdims=True).astype('float32')
        x = np.concatenate((box, conf, j, mask), 1)[conf.reshape(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[np.argsort(-x[:, 4][:max_nms])]
        else:
            x = x[np.argsort(-x[:, 4])]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # with TIMER('nms_2'):
        i = nms_vanil(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]

        # timer limits
        if (time.time() - t) > time_limit:
            LOGGER.warning("NMS time limit exceeded!")
            break  # time limit exceeded
    return output



def nms_vanil(boxes, scores, threshold):
    # vanilla nms

    boxes_area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    keep_indices = []
    indices = scores.argsort()[::-1]
    while indices.size > 0:
        i = indices[0]
        keep_indices.append(i)
        w = np.maximum(0, np.minimum(boxes[:, 2][i], boxes[:, 2][indices[1:]]) - np.maximum(boxes[:, 0][i], boxes[:, 0][indices[1:]]))
        h = np.maximum(0, np.minimum(boxes[:, 3][i], boxes[:, 3][indices[1:]]) - np.maximum(boxes[:, 1][i], boxes[:, 1][indices[1:]]))
        intersection = w * h
        ious = intersection / (boxes_area[i] + boxes_area[indices[1:]] - intersection) 
        indices = indices[np.where(ious <= threshold)[0] + 1]
    return np.asarray(keep_indices)






def pysizeof(obj, seen=None):
    """Recursively finds size of objects in bytes"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += pysizeof(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((pysizeof(v, seen) for v in obj.values()))
        size += sum((pysizeof(k, seen) for k in obj.keys()))
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        try:
            size += sum((pysizeof(i, seen) for i in obj))
        except TypeError:
            logging.exception("Unable to get size of %r. This may lead to incorrect sizes. Please report this error.", obj)
    if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
        size += sum(pysizeof(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))
        
    return size


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern='[|@#!¡·$€%&()=?¿^*;:,¨´><+]', repl='_', string=s)



def setup_logging_plain(stream_logger_name=None, 
                        stream_level=logging.DEBUG,
                        ):

    # stream logger 
    stream_logger = logging.getLogger(stream_logger_name)
    stream_logger.setLevel(stream_level)

    stream_handler = logging.StreamHandler() 
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    stream_handler.setLevel(stream_level)
    stream_logger.addHandler(stream_handler)  

    return stream_logger


def setup_logging(stream_logger_name=None, 
                    stream_level=logging.DEBUG,
                    stream_handler='rich',
                    file_logger_name=None,
                    file_level=logging.DEBUG,
                    log_dir='logs',
                    freq='day'
                    ):

    # stream logger 
    stream_logger = logging.getLogger(stream_logger_name)
    stream_logger.setLevel(stream_level)

    if stream_handler.lower() == 'rich':    # rich handler
        from rich.logging import RichHandler
        stream_handler = RichHandler()
        stream_handler.setFormatter(logging.Formatter("[%(levelname)s] [%(filename)s] [%(lineno)s] %(message)s"))
    elif stream_handler.lower() == 'default':    # logging handler
        stream_handler = logging.StreamHandler() 
        # stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
    stream_handler.setLevel(stream_level)
    stream_logger.addHandler(stream_handler)  


    # file handler
    file_logger = logging.getLogger(file_logger_name)   # TODO: share name?
    file_logger.setLevel(file_level)
    
    # create dir
    if not Path(log_dir).exists():
        Path(log_dir).mkdir()

    # frequency
    if freq.lower() in ('seconds', 'second', 'sec', 's'):
        fm = "%Y_%m_%d_%H_%M_%S"
    elif freq.lower() in ('minute', 'm', 'min', 'minutes'):
        fm = "%Y_%m_%d_%H_%M"
    elif freq.lower() in ('hours', 'hour', 'h'):
        fm = "%Y_%m_%d_%H"
    elif freq.lower() in ('days', 'day', 'd'):
        fm = "%Y_%m_%d"
    else:
        fm = "%Y_%m"

    file_handler = logging.FileHandler(Path(log_dir) / (datetime.now().strftime(fm) + '.log') )
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s(%(lineno)s)] %(message)s'))
    file_handler.setLevel(file_level)
    file_logger.addHandler(file_handler)

    return stream_logger, file_logger
