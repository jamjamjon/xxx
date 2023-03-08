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
from flask import Flask, request, jsonify
import base64
import uuid
import weakref
import time


# path
FILE = Path(__file__).resolve() # file resolve path
ROOT_DIR = FILE.parents[3]  # ROOT_dir
PARENT_DIR = FILE.parents[0]    # parent dir 


# add to python path
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR)) 

# modules 
from core.base_model import EnsembleModel
from core.utils import parse_model_config, CONSOLE, TIMER, gpu_info, resource_info, INSPECTOR, Visualizer, Colors
from core.dataloader import DataLoader  # LoadImages, LoadBatchImages, LoadStreams, 




# yaml path (modify this for every project)
CONFIG_DIR = str(PARENT_DIR / "configs")
CONFIG_NAME = "default"



class MaskWearingModel(EnsembleModel):

    def post_process(self, x):

        head_detector = self.models['head_detector']
        face_mask_unknown_classifier = self.models['face_mask_unknown_classifier']


        ys = head_detector(x)  # head model infer
        # rich.print(ys)

        # rich.print(f"head_detector: {head_detector.ims.shape}")
        # rich.print(f"head_detector: {len(head_detector.im0s)}")


        res = []  # save all results
        for idx, y in enumerate(ys):   # cope with each image results

            # if y is not None and len(y) > 0:  # has objects
            if len(y) > 0:  # has objects

                for *xyxy, conf, cls in reversed(y):
                    # min bbox restrict & class filter 
                    if min(int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])) < head_detector.min_bbox_size:
                        continue

                    # slice filter, optional
                    if max((xyxy[3] - xyxy[1]), (xyxy[2] - xyxy[0])) / min((xyxy[3] - xyxy[1]), (xyxy[2] - xyxy[0])) > 2.3:  
                        continue

                    xyxy = np.asarray(xyxy)  # to ndarray

                    # scale bbox
                    scale_w, scale_h, scale_top = 1.2, 1.15, 0.0   # 1.15, 1.1, 0.0
                    w, h = (xyxy[2] - xyxy[0]), (xyxy[3] - xyxy[1])
                    xyxy[0] -= (scale_w - 1) * w / 2    # left
                    xyxy[2] += (scale_w - 1) * w / 2    # right
                    xyxy[1] -= (scale_h - 1) * scale_top * h / 2    # top
                    xyxy[3] += (scale_h - 1) * (1 - scale_top) * h / 2    # bottom

                    # clip
                    xyxy[[0, 2]] = xyxy[[0, 2]].clip(0, head_detector.im0s[idx].shape[1])
                    xyxy[[1, 3]] = xyxy[[1, 3]].clip(0, head_detector.im0s[idx].shape[0])

                    # crop img
                    crop_img = head_detector.im0s[idx][int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), ::1]  # opencv, BGR

                    # secondary classifier
                    y_cls = face_mask_unknown_classifier([crop_img])[0]   # input one image, take one result
                    # y_cls_idx, y_cls_name, y_cls_conf = y_cls['class_idx'], y_cls['class_name'], y_cls['conf']
                    if len(y_cls) > 0:
                        y_cls_idx, y_cls_name, y_cls_conf = y_cls[0], y_cls[1], y_cls[2]
                    else:
                        continue

                    # rich.print(f"face_mask_unknown_classifier: {face_mask_unknown_classifier.ims.shape}")
                    # rich.print(f"face_mask_unknown_classifier: {len(face_mask_unknown_classifier.im0s)}")


                    # classifier confidence filter
                    if float(y_cls_conf) < face_mask_unknown_classifier.conf_threshold:
                        # print(f' < conf, filtered!')
                        continue

                    # save results
                    res.append({
                        'image_id': idx,
                        # 'image_path': x[idx],
                        'class_idx': int(y_cls_idx),
                        'class_label': y_cls_name,
                        'conf': float(y_cls_conf),
                        'box': list(map(float, xyxy))}
                    )


            else:   # None 
                res.append({
                    'image_id': idx, 
                    # 'image_path': x[idx], 
                    'class_idx': None,
                    'class_label': None, 
                    'conf': None, 
                    'box': None}
                )
 
        return res



# # testing fine
# @INSPECTOR('warmup', repeat=10)
# def warmupppp(model):
#     # t0 = time.time()
#     model.warmup(times=2)
#     # print(f'111 ---> {time.time() - t0}')



# ----------------------------------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg) -> None:
    # print(OmegaConf.to_yaml(cfg))
    cfg = parse_model_config(cfg)
    # rich.print(cfg)

    model = MaskWearingModel(cfg, verbose=False, do_warmup=False)
    # model2 = MaskWearingModel(cfg, verbose=False)  # test
    # model3 = MaskWearingModel_2(cfg, verbose=False)  # test
    # model4 = MaskWearingModel_2(cfg, verbose=False)  # test
    
    # rich.print(EnsembleModel.INSTANCES)
    # del model2, model3
    # rich.print(EnsembleModel.INSTANCES)
    # a = model2
    # b = a

    # del model3, model4
    # model.instances_info()   # bugs
    # CONSOLE.print(EnsembleModel.get_instances_table())
    # CONSOLE.print(MaskWearingModel.get_instances_table())


    # test device switch
    # # print(model.device)
    # model.to('cpu')
    # CONSOLE.print(EnsembleModel.get_instances_table())

    # # print(model.device)
    # model.to(1)
    # CONSOLE.print(EnsembleModel.get_instances_table())

    # # print(model.device)

    # model.to(99)
    # CONSOLE.print(EnsembleModel.get_instances_table())
    # # print(model.device)


    # resource_info(display=True)
    # gpu_info(display=True)

    # test model warm up time
    # for x in range(10):
    #     with INSPECTOR('warmup'):
    #         model.warmup(3)

    # model.warmup(batch_size=3)   # do first
    # rich.print(MaskWearingModel.get_instances_table())

    # img = [
    #         "deploy/assets/faces.jpg",
    #         "deploy/assets/0.jpg",
    #         "deploy/assets/7.jpg",
    #     ]


    # ----------------------------
    #   multi input data infer  
    # ----------------------------
    source = [
            # "deploy/assets/0.jpg",
            # "deploy/assets/7.jpg",
            # "deploy/assets/faces.jpg",
            # "deploy/assets/faces.jpg",  # auto remove 
            # "/home/zhangj/MLOPS/MODELS/AutoInfer/xxx-master/deploy/assets/playing_phone.mp4",

            # rtsp should not input with above
            "rtsp://admin:zfsoft888@192.168.0.127:554/h265/ch1/",
            "rtsp://admin:KCNULU@192.168.3.107:554/h264/ch1/",
            # "http://devimages.apple.com.edgekey.net/streaming/examples/bipbop_4x3/gear2/prog_index.m3u8",
            ]

    dataloader = DataLoader(source=source, batch_size=2, vid_stride=10)
    # dataloader = DataLoader(source='deploy/projects/playing_phone/rtsp.txt')
    # dataloader = DataLoader(source='deploy/projects/playing_phone/source.txt', batch_size=2)

    vis = Visualizer(line_width=2, color=Colors(shuffle=False)(1))   # 

    # ----------------------------
    #   save result
    # ----------------------------
    # vid_path, vid_writer = [None] * bs, [None] * bs
    # save_dir = 'runs' 
    # if not Path(save_dir).exists():
    #     Path(save_dir).mkdir()



    for idx, (path, im0, vid_cap, msg) in enumerate(dataloader):
        with TIMER(f"{idx}-th:"):
            y = model(im0)    # infer   im0: [img, img, ...]
        rich.print(y)


        # visualize
        for idx, elem in enumerate(y):
            image_id = elem['image_id']

            if elem['box']:
                xyxy = elem['box']
                conf = elem['conf']
                class_label = elem['class_label']
                vis.draw(im0[image_id], box=xyxy, label=class_label, conf=f"{conf:.4f}")  # draw


            cv2.imshow('demo_' + str(image_id), im0[image_id])
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                return




    # with INSPECTOR('========>'):
    #     y = model(img)
        # rich.print(y)


    # img2 = [
    #         "deploy/assets/0.jpg",
    #         "deploy/assets/0.jpg",
    #         "deploy/assets/0.jpg",
    #         # "deploy/assets/faces.jpg",
    #         # "deploy/assets/7.jpg",
    #         # "deploy/assets/7.jpg",
    #         # "deploy/assets/7.jpg",
    #     ]


    # model.to(1)  # warmup automatically
    # rich.print(MaskWearingModel.get_instances_table())
    # with INSPECTOR('batch=3'):
    #     y = model(img2)
        # rich.print(y)


    # for i in range(10):
    #     with TIMER(f"{i}-th"):
    #         y = model(img)


    # # visualize
    # for i, res in enumerate(y):
    #     image_id = res['image_id']
    #     class_label = res['class_label']
    #     xyxy = res['box']


    #     image = cv2.imread(img[image_id])

    #     cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2, cv2.LINE_AA)  # filled
    #     cv2.putText(image, str(class_label),
    #                 (int(xyxy[0]), int(xyxy[1] - 5)), 0, 0.7, 
    #                 (0, 255, 0), 
    #                 thickness=1, lineType=cv2.LINE_AA)


    #     cv2.imshow('demo', image)
    #     cv2.waitKey(0)



    # for i in range(10):
    #     # with TIMER(f'{i}_th time'):
    #     y = model(img)


    # # each model module test
    # model.models['head_detector'].test()
    # model.models['face_mask_unknown_classifier'].test()



# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
