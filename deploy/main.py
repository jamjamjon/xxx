import numpy as np
from pathlib import Path
import cv2
import sys
import time
from omegaconf import DictConfig, OmegaConf
import hydra
import rich
from rich.progress import track as ProgressBar
from flask import Flask, request, jsonify
import base64
import weakref
import uuid
import psutil
from datetime import datetime


# ---------------------------------------------------------------------------
FILE = Path(__file__).resolve() # file resolve path
ROOT_DIR = FILE.parents[1]  # ROOT_dir
PARENT_DIR = FILE.parents[0]    # parent dir 

# add to python path
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR)) 

# [sys.path.append(str(x)) for x in list(Path(__file__).resolve().parents) if str(x) not in sys.path]


# modules 
from core.utils import (parse_model_config, get_projects_configs, FLASK_APP, CONSOLE, LOGGER, build_resolver,
                        gpu_info)
from core.base_model import EnsembleModel
from deploy.all_rounder import AllRounder


# yaml path (modify this for every project)
CONFIG_DIR = str(PARENT_DIR / "projects")
CONFIG_NAME = "projects_default"

# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    # CONSOLE.print(OmegaConf.to_yaml(cfg), '\n\n')
    
    gpu_info()

    build_resolver() # build resolver
    all_models_configs = get_projects_configs(configs=cfg)  # parse projects configs   

    rich.print(all_models_configs)
    exit()


    all_rounder = AllRounder(all_models_configs=all_models_configs, tracking_class=EnsembleModel, verbose=False)  # init profiler
    all_rounder.requests_solver()   # start 




# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
    FLASK_APP.run(host='127.0.0.1', port=10612, threaded=True)

