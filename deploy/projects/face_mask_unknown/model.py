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
from core.utils import parse_model_config


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
                     'class_idx:': int(y[0]), 
                     'class_name:': y[1], 
                     'conf': float(y[2])}
                )
            else:
                res.append({
                    'image_id': idx, 
                    # 'image_path': x[idx], 
                    'class_name:': None, 
                    'class_idx:': None, 
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
    model2 = FaceMaskModel(cfg, verbose=False, do_warmup=False)


    a = model 
    b = model


    del model2

    rich.print(FaceMaskModel.get_instances_table())
    # model.instances_info()


    img = [
            "deploy/assets/1.jpg",
            "deploy/assets/2.jpg",
            "deploy/assets/0.jpg"
        ]
    y = model(img)    # infer
    rich.print(y)  # results


    img2 = [
            "deploy/assets/0.jpg",
            "deploy/assets/0.jpg"
         ]


    model.to(1)
    rich.print(FaceMaskModel.get_instances_table())
    y=model(img2)
    rich.print(y)




    # test
    # y = model.models['face_mask_unknown'].test()



if __name__ == '__main__':
    main()
