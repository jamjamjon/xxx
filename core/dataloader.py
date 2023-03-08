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
# from core.utils import letterbox, IMG_FORMAT, VIDEO_FORMAT, LOGGER, 
from core.utils import (WEIGHTS_TYPE, MODEL_TYPE, IMG_FORMAT,VIDEO_FORMAT, STREAM_FORMAT, 
                        CenterCrop, Softmax, Normalize, letterbox, parse_model_config, CONSOLE, 
                        MB, GB, KB, pysizeof, LOGGER, INSPECTOR, InstanceInfo, DeviceInfo, clean_str,
                        scale_boxes_batch, xywh2xyxy, TIMER, batched_nms, gpu_info, resource_info,
                        )



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
                            f'Supported formats are:\nimages: {IMG_FORMAT}'

        self.path = path




class DataLoader:
    # only load images to numpy.ndarry, Don't do any pre-process here.
    # return im0: ndarray, list | no stacking, images may not has same shape, do it in Ensemable
    # stack(pre-process(x)) in EnsemableModel()
    # modes:
    # > images mode: batch infer & frame by frame infer. input type: txt file and list()
    # > videos mode: frame by frame infer. input type: txt file and list()
    # > streams mode: batch infer with multi-stream, but frame by frame at each stream. input type: txt file and list()

    def __init__(self, source, vid_stride=1, batch_size=2):
        self.vid_stride = vid_stride    # video frame-rate stride
        self.batch_size = batch_size    # only works at 'images' mode

        # txt file
        p_images, p_videos, p_files, p_streams = [], [], [], []   # save image_path, video_path, files, streams
        if isinstance(source, str) and Path(source).suffix == '.txt':  # *.txt file -> list
            source = Path(source).read_text().split()

        # dedupilicated source
        self.source = list(set(source))

        # all input data 
        for x in tqdm(self.source, 'DataLoader loading...'):
            if x.lower().startswith(STREAM_FORMAT):  # url
                p_streams.append(str(x))
            elif Path(x).suffix in IMG_FORMAT:   # image
                p_images.append(str(Path(x).resolve()))
            elif Path(x).suffix in VIDEO_FORMAT:  # video
                p_videos.append(str(Path(x).resolve()))
            elif Path(x).is_dir():    # dir
                p_images.extend([str(Path(elem).resolve()) for elem in Path(x).rglob('**/*') if elem.suffix in IMG_FORMAT])
                p_videos.extend([str(Path(elem).resolve()) for elem in Path(x).rglob('**/*') if elem.suffix in VIDEO_FORMAT])
            else:
                LOGGER.error(f"`{x}` is not support for now!")
                sys.exit(-1)

        # dedupilicated again
        self.p_images, self.p_videos, self.p_streams = map(lambda x: list(set(x)), (p_images, p_videos, p_streams))
        self.p_files = self.p_images + self.p_videos

        # number
        self.num_files = len(self.p_files)
        self.num_images = len(self.p_images)
        self.num_videos = len(self.p_videos)
        self.num_streams = len(self.p_streams)

        assert self.num_files > 0 or self.num_streams > 0, f"No input data!" \
            f"Supported formats are:\n> images: {IMG_FORMAT}\n> videos: {VIDEO_FORMAT}\n> streams: {STREAM_FORMAT}"

        # streams & images_videos
        assert not all((len(self.p_files) > 0, len(self.p_streams) > 0)), f"ERROR: has `image/video` and `stream` both in {self.source}"

        # mode
        if len(self.p_streams) > 0:
            self.mode = 'streams'  #  streams: batch infer       
        elif len(p_videos) > 0:
            self.mode = 'videos'   # images * videos
        else:
            self.mode = 'images'  # pure images: batch infer
        LOGGER.info(f"Mode -> {self.mode}")

        # videos mode
        if self.mode == 'videos':
            self.video_flag = [False] * self.num_images + [True] * self.num_videos
            if any(self.p_videos):
                self._new_video(self.p_videos[0])  # new video
            else:
                self.cap = None

        # streams mode
        elif self.mode == 'streams': 
            self.imgs = [None] * self.num_streams
            self.fps = [0] * self.num_streams
            self.frames = [0] * self.num_streams
            self.threads = [None] * self.num_streams

            for i, s in enumerate(self.p_streams):  # index, source
                # Start thread to read frames from video stream

                st = f'{i + 1}/{self.num_streams}: {s}... '
                s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
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
            # s = np.stack([x for x in self.imgs])  # stacking images

        # images mode
        elif self.mode == 'images':
            self.b0 = 0
            self.num_batches = int((self.num_images + self.batch_size - 1) / self.batch_size)     # batch infer, ceil(num_batch)


    def __len__(self):
        if self.mode == 'videos':
            return self.num_files   # number of files
        elif self.mode == 'streams':
            return self.num_streams     # number of streams
        elif self.mode == 'images':
            return self.num_batches   # number of image batches


    def __iter__(self):
        self.count = 0
        return self


    def __next__(self):
        if self.mode == 'videos':   # video mode
            if self.count == self.num_files:
                raise StopIteration
            path = self.p_files[self.count]

            # videos
            if self.video_flag[self.count]:   
                for _ in range(self.vid_stride):
                    self.cap.grab()
                ret_val, im0 = self.cap.retrieve()
                while not ret_val:
                    self.count += 1   # count +1
                    self.cap.release()
                    if self.count == self.num_files:  # reach last
                        raise StopIteration
                    path = self.p_files[self.count]  # move to next
                    self._new_video(path)   # new video capture
                    ret_val, im0 = self.cap.read()
                self.frame += 1  # frame +1
                # info
                msg = f'[Videos Mode] Video {self.count + 1}/{self.num_files} ({self.frame}/{self.frames}) {path}: '
            # images
            else:   
                self.count += 1
                im0 = cv2.imread(path)  
                assert im0 is not None, f"> Image Not Found {path} !"
                msg = f"[Videos Mode] Image {self.count}/{self.num_files} {path}: "

            return path, [im0], self.cap, msg  # image pre-process will do within Ensemable


        elif self.mode == 'streams':  # stream mode
            self.count += 1

            if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
                cv2.destroyAllWindows()
                raise StopIteration

            im0 = self.imgs.copy()  # TODO 
            # im = np.stack([x for x in im0])  # stacking images for batch infer
            msg = f"[Streams Mode] Stacked Image {self.count}/{self.frames[0]} {self.p_streams}: "
            return self.p_streams, im0, None, msg
            

        elif self.mode == 'images':  # images
            if self.count == self.num_batches:
                raise StopIteration

            self.count += 1
            b1 = self.num_images if self.b0 + self.batch_size > self.num_images else self.b0 + self.batch_size
            batch_img = self.p_images[self.b0: b1]
            im0 = [cv2.imread(x) for x in batch_img]
            self.b0 = b1
            msg = f"[Images Mode] Image(bs={self.batch_size}) {self.count}/{self.num_batches} {batch_img}: "
            return batch_img, im0, None, msg



    def _new_video(self, path):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)


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
                    LOGGER.warning('Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(0.0)  # wait time



