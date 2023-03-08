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
from hydra.core.hydra_config import HydraConfig

# path
FILE = Path(__file__).resolve() # file resolve path
ROOT_DIR = FILE.parents[3]  # ROOT_dir
PARENT_DIR = FILE.parents[0]    # parent dir 


# add to python path
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR)) 

# modules 
# from core.base_model import BaseClassifier
from core.base_model import EnsembleModel
from core.utils import parse_model_config, Visualizer, Colors, TIMER
from core.dataloader import DataLoader  # LoadImages, LoadBatchImages, LoadStreams, 


# yaml path (modify this for every project)
CONFIG_DIR = str(PARENT_DIR / "configs")
CONFIG_NAME = "default"



# initialize instance
class FaceMaskModel(EnsembleModel):

    # you must inplement
    def post_process(self, x):
        # x must be image list => ['', '', ]

        face_mask_model = self.models['face_mask_unknown']  # playing phone model
        ys = face_mask_model(x)  # infer
        # rich.print(ys)

        # batch infer
        res = []
        for idx, y in enumerate(ys): 
            if y is not None:
                res.append(
                    {'image_id': idx, 
                     # 'image_path': x[idx], 
                     'class_idx': int(y[0]), 
                     'class_name': y[1], 
                     'conf': float(y[2])}
                )
            else:
                res.append({
                    'image_id': idx, 
                    # 'image_path': x[idx], 
                    'class_name': None, 
                    'class_idx': None, 
                    'conf': None}
                )

        return res


# ----------------------------------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg):
    # print(OmegaConf.to_yaml(cfg))

    cfg = parse_model_config(cfg)

    # rich.print(cfg)

    model = FaceMaskModel(cfg, verbose=False, do_warmup=False)





    # ----------------------------
    #   multi input data infer  
    # ----------------------------
    source = [
            "deploy/assets/0.jpg",
            "deploy/assets/1.jpg",
            "deploy/assets/2.jpg",
            # "/home/zhangj/MLOPS/MODELS/AutoInfer/xxx-master/deploy/assets/playing_phone.mp4",

            # rtsp should not input with above
            # "rtsp://admin:zfsoft888@192.168.0.127:554/h265/ch1/",
            # "rtsp://admin:KCNULU@192.168.3.107:554/h264/ch1/",
            # "http://devimages.apple.com.edgekey.net/streaming/examples/bipbop_4x3/gear2/prog_index.m3u8",
            ]


    dataloader = DataLoader(source=source, batch_size=1, vid_stride=10)   # classifier batch size must be 1
    # dataloader = DataLoader(source='deploy/projects/playing_phone/rtsp.txt')
    # dataloader = DataLoader(source='deploy/projects/playing_phone/source.txt', batch_size=2)

    vis = Visualizer(line_width=2, color=Colors(shuffle=False)(1))   # visualize



    for idx, (path, im0, vid_cap, msg) in enumerate(dataloader):
        with TIMER(f"{idx}-th:"):
            y = model(im0)    # infer   im0: [img, img, ...]
        rich.print(y)

        # visualize
        for idx, elem in enumerate(y):


            image_id = elem['image_id']

            if elem['class_name']:
                conf = elem['conf']
                class_label = elem['class_name']
                vis.draw(im0[image_id], label=class_label, conf=f"{conf:.4f}")  # draw


        cv2.imshow('demo_' + str(image_id), im0[image_id])
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            return











    # model2 = FaceMaskModel(cfg, verbose=False, do_warmup=False)


    # a = model 
    # b = model


    # del model2

    # rich.print(FaceMaskModel.get_instances_table())
    # # model.instances_info()


    # img = [
    #         "deploy/assets/1.jpg",
    #         "deploy/assets/2.jpg",
    #         "deploy/assets/0.jpg"
    #     ]
    # y = model(img)    # infer
    # rich.print(y)  # results


    # img2 = [
    #         "deploy/assets/0.jpg",
    #         "deploy/assets/0.jpg"
    #      ]


    # model.to(1)
    # rich.print(FaceMaskModel.get_instances_table())
    # y=model(img2)
    # rich.print(y)




    # test
    # y = model.models['face_mask_unknown'].test()



if __name__ == '__main__':
    main()
