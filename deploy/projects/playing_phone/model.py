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
                        get_device)


# yaml path (modify this for every project)
CONFIG_DIR = str(PARENT_DIR / "configs")
CONFIG_NAME = "default"


# derived model class
class PlayingPhoneModel(EnsembleModel):

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


    # print(get_device(1))
    # print(get_device('cuda: 999'))
    # print(get_device(' cuda:0 '))
    # print(get_device('1'))
    # print(get_device('cpu'))


    model = PlayingPhoneModel(cfg, verbose=False, do_warmup=False)

    rich.print(model.get_instances_table())


    # model.instances_info()

    # rich.print(model.instances)


    # model.to('cpu')
    # rich.print(model.get_instances_table())

    # model.to(2)
    # rich.print(model.get_instances_table())

    # exit()
    # model2 = PlayingPhoneModel(cfg, verbose=False, do_warmup=False)
    # model3 = PlayingPhoneModel(cfg, verbose=False, do_warmup=False)
    # model4 = PlayingPhoneModel(cfg, verbose=False, do_warmup=False)


    # a = model2
    # b = model2

    # del model3, model4

    # model.instances_info()   # has bugs , dont user this
    # rich.print(PlayingPhoneModel.get_instances_table())




    # resource_info(display=True)
    # gpu_info(display=True)



    # batch infer
    img = [
            "deploy/assets/5.jpg",
            "deploy/assets/6.jpg",
            "deploy/assets/bus.jpg",
            ]

    y = model(img)    # infer
    rich.print(y)


    # @Dataset
    # y = model(img)







    sys.exit()

    img2 = [
            "deploy/assets/5.jpg",
            "deploy/assets/5.jpg",
         ]


    model.to(0)
    rich.print(PlayingPhoneModel.get_instances_table())
    y=model(img2)
    rich.print(y)


    

    # model.to('cpu')
    # rich.print(PlayingPhoneModel.get_instances_table())
    # y=model(img2)
    # rich.print(y)




    
    # model.to('cpu')
    # rich.print(PlayingPhoneModel.get_instances_table())
   
    exit()
    for i in range(5):
        rich.print(PlayingPhoneModel.get_instances_table())

        with TIMER(f"{i}-th:"):
            y = model(img)
            #rich. print(y)
            model.to(1)  #   ==============> may has bug!!

            # print('-==================\n')

    # # visualize
    # for i, res in enumerate(y):
    #     image_id = res['image_id']
    #     class_label = res['class_label']
    #     xyxy = res['box']


    #     image = cv2.imread(img[image_id])

    #     if xyxy is not None:

    #         cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2, cv2.LINE_AA)  # filled
    #         cv2.putText(image, str(class_label),
    #                     (int(xyxy[0]), int(xyxy[1] - 5)), 0, 0.7, 
    #                     (0, 255, 0), 
    #                     thickness=1, lineType=cv2.LINE_AA)


    #     cv2.imshow('demo', image)
    #     cv2.waitKey(0)




    
    # model.models['playing_phone'].test()    # test



if __name__ == '__main__':
    main()

