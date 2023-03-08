import tensorrt

import numpy as np
import onnxruntime
from pathlib import Path
import cv2
import sys
import yaml
import rich
from omegaconf import DictConfig, OmegaConf
import hydra
from abc import ABC, abstractmethod
import rich
import time
from flask import Flask, request, jsonify
import base64



# path
FILE = Path(__file__).resolve() # file resolve path
ROOT_DIR = FILE.parents[3]  # ROOT_dir
PARENT_DIR = FILE.parents[0]    # parent dir 

# add to python path
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR)) 

# modules 
from core.base_model import EnsembleModel
from core.utils import (parse_model_config, CONSOLE, build_resolver, resource_info, gpu_info, TIMER, 
                        get_device, Visualizer, Colors)
from core.dataloader import DataLoader  # LoadImages, LoadBatchImages, LoadStreams, 



# yaml path (modify this for every project)
CONFIG_DIR = str(PARENT_DIR / "configs")
CONFIG_NAME = "default"


# derived model class
class PlayingPhoneModel(EnsembleModel):

    # @DataLoader()
    # def __call__(self, x):
    #     return self.post_process(x)


    # you must inplement
    def post_process(self, x):
        # x => image list

        model = self.models['playing_phone']  # playing phone model
        ys = model(x)   # infer  [(num_object, 6), (num_object, 6)]

        # rich.print(ys)

        # visulize 
        # model.visulize(model.im0, xyxys=x[:, 0:4], conf=x[:, 4])  # draw to test

        # batch size > 1

        # always has results
        # if len(y) > 0 and len(y[0]) > 0:  # got pred
        
        res = []  # save all results
        for idx, y in enumerate(ys):   # cope with each image results

            # if y is not None and len(y) > 0:
            if len(y) > 0:
                # seprate multi-classes boxes
                dets = []
                for c in np.unique(y[:, -1]):   # cellphone, cellphone-holding
                    dets.append(y[y[:, -1] == c])

                # make sure has two-classes boxes
                if len(dets) == 2:
                    for *xyxy_cellphone, conf_cellphone, cls_cellphone in reversed(dets[0]):
                        for *xyxy_holding, conf_holding, cls_holding in reversed(dets[1]):

                            # overlap
                            w = max(min(xyxy_holding[2], xyxy_cellphone[2]) - max(xyxy_holding[0], xyxy_cellphone[0]), 0)
                            h = max(min(xyxy_holding[3], xyxy_cellphone[3]) - max(xyxy_holding[1], xyxy_cellphone[1]), 0)
                            overlap = w * h

                            # overlap / cellphone area
                            self_iou = overlap / ((xyxy_cellphone[2] - xyxy_cellphone[0]) * (xyxy_cellphone[3] - xyxy_cellphone[1])) 

                            # threshold
                            if self_iou > model.overlap_self_iou:

                                # save result
                                res.append(
                                    {'image_id': idx,
                                     # 'image_path': x[idx],
                                     'class_idx': int(cls_holding),
                                     'class_label': model.classes_names[int(cls_holding)],
                                     'conf': float(conf_holding),
                                     'box': list(map(float, xyxy_holding))}
                                )

                else:  # only has one class
                    res.append({
                        'image_id': idx, 
                        # 'image_path': x[idx], 
                        'class_idx': None, 
                        'class_label': None, 
                        'conf': None, 
                        'box': None}
                    )
            else:
                res.append({
                    'image_id': idx, 
                    # 'image_path': x[idx], 
                    'class_idx': None, 
                    'class_label': None, 
                    'conf': None, 
                    'box': None}
                )
 
        return res




# ----------------------------------------------------------------------------------------------------
# @hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="maskwearing")
@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg) -> None:
    build_resolver()
    # print(OmegaConf.to_yaml(cfg))

    # context initialization
    # with initialize(config_path="conf", job_name="test_app"):
    #     cfg = compose(config_name="config", overrides=["db=mysql", "db.user=me"])
    #     print(OmegaConf.to_yaml(cfg))

    # global initialization
    # initialize(config_path="conf", job_name="test_app")
    # cfg = compose(config_name="config", overrides=["db=mysql", "db.user=me"])
    # print(OmegaConf.to_yaml(cfg))


    cfg = parse_model_config(cfg)
    # rich.print(cfg)
    model = PlayingPhoneModel(cfg, verbose=False, do_warmup=False)
    # rich.print(model.get_instances_table())   # model.instances_info()
    # rich.print(model.instances)

    # ----------------------------
    #   switch device test    
    # ----------------------------
    # model.to('cpu')
    # rich.print(model.get_instances_table())

    # model.to(2)
    # rich.print(model.get_instances_table())


    # ----------------------------
    #   multi instances test    
    # ---------------------------- 
    # model2 = PlayingPhoneModel(cfg, verbose=False, do_warmup=False)
    # model3 = PlayingPhoneModel(cfg, verbose=False, do_warmup=False)
    # model4 = PlayingPhoneModel(cfg, verbose=False, do_warmup=False)
    # a = model2
    # del model3, model4, model2
    # rich.print(PlayingPhoneModel.get_instances_table())     # model.instances_info()   # has bugs , dont user this


    # ----------------------------
    #   cpu & gpu info    
    # ---------------------------- 
    # resource_info(display=True)
    # gpu_info(display=True)


    # ----------------------------
    #   multi input data infer  
    # ---------------------------- 
    source = [
            "deploy/assets/0.jpg",
            "deploy/assets/5.jpg",
            "deploy/assets/6.jpg",
            "deploy/assets/bus.jpg",
            # "deploy/assets/bus.jpg",  # auto remove 
            # "/home/zhangj/MLOPS/MODELS/AutoInfer/xxx-master/deploy/assets/playing_phone.mp4",

            # # rtsp should not input with above
            # "rtsp://admin:zfsoft888@192.168.0.127:554/h265/ch1/",
            # "rtsp://admin:KCNULU@192.168.3.107:554/h264/ch1/",
            # "http://devimages.apple.com.edgekey.net/streaming/examples/bipbop_4x3/gear2/prog_index.m3u8",
            ]

    dataloader = DataLoader(source=source, batch_size=2, vid_stride=1)
    # dataloader = DataLoader(source='deploy/projects/playing_phone/rtsp.txt')
    # dataloader = DataLoader(source='deploy/projects/playing_phone/source.txt', batch_size=2)

    vis = Visualizer(line_width=2, color=Colors(shuffle=False)(1))   # 

    for idx, (path, im0, vid_cap, msg) in enumerate(dataloader):
        with TIMER(f"{idx}-th:"):
            y = model(im0)    # infer   im0: [img, img, ...]
        # rich.print(y)

        # # visualize
        # for idx, elem in enumerate(y):
        #     image_id = elem['image_id']

        #     if elem['box']:
        #         xyxy = elem['box']
        #         conf = elem['conf']
        #         class_label = elem['class_label']
        #         vis.draw(im0[image_id], box=xyxy, label=class_label, conf=f"{conf:.4f}")  # draw


        #     cv2.imshow('demo_' + str(image_id), im0[image_id])
        #     key = cv2.waitKey(1)
        #     if key == 27:
        #         cv2.destroyAllWindows()
        #         return


    # ----------------------------
    #   multi input data infer  
    #   switch device
    # ---------------------------- 
    # source2 = [
    #         # "deploy/assets/5.jpg",
    #         # "deploy/assets/6.jpg",
    #         # "deploy/assets/bus.jpg",
    #         # "deploy/assets/bus.jpg",  # auto remove 
    #         # "/home/zhangj/MLOPS/MODELS/AutoInfer/xxx-master/deploy/assets/playing_phone.mp4",

    #         # rtsp should not input with above
    #         "rtsp://admin:zfsoft888@192.168.0.127:554/h265/ch1/",
    #         "rtsp://admin:KCNULU@192.168.3.107:554/h264/ch1/",
    #         "http://devimages.apple.com.edgekey.net/streaming/examples/bipbop_4x3/gear2/prog_index.m3u8",
    #         ]

    # model.to(0)  # change device
    # rich.print(model.get_instances_table())   # model.instances_info()

    # dataloader2 = DataLoader(source=source2, batch_size=2, vid_stride=10)
    # for idx, (path, im0, vid_cap, msg) in enumerate(dataloader2):
    #     with TIMER(f"{idx}-th:"):
    #         y = model(im0)    # infer   im0: [img, img, ...]
    #     rich.print(y)

    #     # visualize
    #     for idx, elem in enumerate(y):
    #         image_id = elem['image_id']

    #         if elem['box']:
    #             xyxy = elem['box']
    #             conf = elem['conf']
    #             class_label = elem['class_label']
    #             vis.draw(im0[image_id], box=xyxy, label=class_label, conf=f"{conf:.4f}")  # draw


    #         cv2.imshow('demo_' + str(image_id), im0[image_id])
    #         key = cv2.waitKey(1)
    #         if key == 27:
    #             cv2.destroyAllWindows()
    #             return


    
    # model.models['playing_phone'].test()    # test



if __name__ == '__main__':
    main()


