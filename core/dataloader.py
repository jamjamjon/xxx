import contextlib
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
import rich
import numpy as np
import psutil
import yaml
from tqdm import tqdm
import sys
import cv2


# path
FILE = Path(__file__).resolve() # file resolve path
ROOT_DIR = FILE.parents[1]  # ROOT_dir
PARENT_DIR = FILE.parents[0]    # parent dir 


# add to python path
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR)) 




# modules 
from core.utils import letterbox, IMG_FORMAT, VIDEO_FORMAT, LOGGER
from core.utils import (WEIGHTS_TYPE, MODEL_TYPE, IMG_FORMAT, CenterCrop, Softmax, Normalize, letterbox, 
                        parse_model_config, CONSOLE, MB, GB, KB, pysizeof, LOGGER, INSPECTOR,
                        scale_boxes_batch, xywh2xyxy, TIMER, batched_nms, gpu_info, resource_info,
                        InstanceInfo, DeviceInfo, clean_str)






class LoadBatchImages:
    # ['/Users/jamjon/Desktop/xxx/deploy/assets/5.jpg', 
        # '/Users/jamjon/Desktop/xxx/deploy/assets/6.jpg', 
        # '/Users/jamjon/Desktop/xxx/deploy/assets/bus.jpg']


    def __init__(self, path):
        if isinstance(path, str) and Path(path).suffix == '.txt':  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        self.images = [x for x in files if Path(x).suffix.lower() in IMG_FORMAT]
        # self.img_size = img_size
        self.ni = len(self.images)  # number of files
        assert self.ni > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}'

        # rich.print(f'self.images ---> {self.images}')


    def pre_process(self, size, is_detector=False, is_classifier=False, is_other=False):
        # batch images pre-process

        # preprocess
        im_list, im0_list = [], []
        for image in self.images:
            im, im0 = self._pre_process_single_image(
                x=image, 
                size=size,
                is_detector=is_detector, 
                is_classifier=is_classifier, 
                is_other=is_other
            )   # pre-process single image
            im_list.append(im)
            im0_list.append(im0)

        #  batch_images, original_image_list --->  cpu data
        return np.concatenate(im_list, 0), im0_list  # (batch, 3, size, size ), original image list(np.ndarray)


    def _pre_process_single_image(self, x, size, is_detector=False, is_classifier=False, is_other=False):
        if isinstance(x, (str, np.ndarray)):

            # read image if it is not ndarray
            if isinstance(x, str):   
                x = cv2.imread(x)
            assert isinstance(x, np.ndarray), f'Image type should be np.ndarray'

            # pre-process for all types of models
            if is_detector:
                return self._pre_process_detector(x, size)
            elif is_classifier:
                return self._pre_process_classifier(x, size)
            elif is_other:  # others
                return self.pre_process_other(x, size)

        else:
            LOGGER.error(f'Type of element in Input image list is wrong! ---> {x}')
            sys.exit()



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




class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, vid_stride=1):
        if isinstance(path, str) and Path(path).suffix == '.txt':  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()

        # if isinstance(path, (list, tuple)):  # *.txt file with img/vid/dir on each line
        #     path = path

        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:

            p = str(Path(p).resolve())

            if '*' in p:

                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):

                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):

                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')


        images = [x for x in files if Path(x).suffix.lower() in IMG_FORMAT]
        videos = [x for x in files if Path(x).suffix.lower() in VIDEO_FORMAT]
        ni, nv = len(images), len(videos)

        rich.print(f'files ---> {files}')
        rich.print(f'images ---> {images}')
        rich.print(f'videos ---> {videos}')


        # self.img_size = img_size
        # self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        # self.auto = auto
        # self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'


        rich.print(f'self.files ---> {self.files}')
        rich.print(f'self.video_flag ---> {self.video_flag}')


        # sys.exit()


    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        # if self.transforms:
        #     im = self.transforms(im0)  # transforms
        # else:
        #     im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
        #     im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        #     im = np.ascontiguousarray(im)  # contiguous

        # return path, im, im0, self.cap, s
        return path, im0, self.cap, s


    def _new_video(self, path):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        # Rotate a cv2 video manually
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        return self.nf  # number of files


class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='file.streams', vid_stride=1):
        # torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = 'stream'
        self.vid_stride = vid_stride  # video frame-rate stride

        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.sources = [clean_str(x) for x in sources]  # clean source names for later

        print(f"sources --> {sources}")

        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n


        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
                # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/Zgi9g1ksQHc'
                # check_requirements(('pafy', 'youtube_dl==2020.12.2'))
                import pafy
                s = pafy.new(s).getbest(preftype='mp4').url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            # if s == 0:
                # assert not is_colab(), '--source 0 webcam unsupported on Colab. Rerun command in a local environment.'
                # assert not is_kaggle(), '--source 0 webcam unsupported on Kaggle. Rerun command in a local environment.'

            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f'{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)')
            self.threads[i].start()
        LOGGER.info('')  # newline


        # rich.print(len(self.imgs))
        # rich.print(self.imgs[0].shape)
        # rich.print(self.fps)
        # rich.print('frame: ', self.frames)
        # rich.print(self.threads)


        # check for common shapes
        s = np.stack([x for x in self.imgs])
        # s = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0].shape for x in self.imgs])
        # self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        # self.auto = auto and self.rect
        # self.transforms = transforms  # optional
        # if not self.rect:
        #     LOGGER.warning('WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.')


    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f = 0, self.frames[i]  # frame number, frame array
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()  # .read() = .grab() followed by .retrieve()
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(0.0)  # wait time

    def __iter__(self):
        self.count = -1
        rich.print('iter', self.count)

        return self

    def __next__(self):
        self.count += 1

        rich.print('next', self.count)

        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        im0 = self.imgs.copy()
        im = np.stack([x for x in im0])  # resize


        # if self.transforms:
        #     im = np.stack([self.transforms(x) for x in im0])  # transforms
        # else:

            # im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])  # resize
            # rich.print(f'im: {im.shape}')
            

            # im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            # im = np.ascontiguousarray(im)  # contiguous

            # rich.print(f'im: {im.shape}')


        return self.sources, im, im0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years
