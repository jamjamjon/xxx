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
import pynvml



# ---------------------------------------------------------------------------
FILE = Path(__file__).resolve() # file resolve path
ROOT_DIR = FILE.parents[1]  # ROOT_dir
PARENT_DIR = FILE.parents[0]    # parent dir 

# add to python path
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR)) 

# modules 
from core.utils import parse_model_config, get_projects_configs, FLASK_APP, CONSOLE, TIMER, INSPECTOR, MB, KB, GB
from core.base_model import EnsembleModel


# ----------------------------------------------
# projects model class
# TODO: add here when has new model class
# ----------------------------------------------
from deploy.projects.playing_phone.model import PlayingPhoneModel
from deploy.projects.mask_wearing.model import MaskWearingModel
from deploy.projects.face_mask_unknown.model import FaceMaskModel


# yaml path (modify this for every project)
CONFIG_DIR = str(PARENT_DIR / "projects")
CONFIG_NAME = "default"
# ---------------------------------------------



class AllRounder:
    # all-rounder

    def __init__(self, all_models_configs, tracking_class=EnsembleModel, verbose=True):
        self.all_models_configs = all_models_configs
        self.tracking_class = tracking_class
        self.verbose = verbose
        
        self.flask_url = '/v1.0'
        self.add_model_field = 'add'
        self.remove_model_field = 'remove'
        self.image_field = 'image'


    @TIMER(prefix='Update function')
    def submit(self, add: list=None, remove: list=None, warmup=False, verbose=False):

        # add models
        if add is not None:
            for x in add:

                # already has one instance
                if x in self.tracking_class.INSTANCES['active'].keys():
                    continue

                # build model from all models configs
                if x in self.all_models_configs.keys():
                    cfg = self.all_models_configs[x]  # get config

                    # build 
                    cfg['model_instance_name'] = eval(cfg['model_class'])(config=parse_model_config(cfg['model_config']), 
                                                                          verbose=verbose,
                                                                          do_warmup=warmup) 
                else:   # no component name
                    CONSOLE.log(f"[add warning] No `{x}` component, please check the component name!")

            CONSOLE.log(f"Done add models.")


        # remove models
        if remove is not None:
            for x in remove:

                # no this component name
                if x not in self.all_models_configs.keys():
                    CONSOLE.log(f"No `{x}` component name, please check the component name!")
                    continue

                # no in active
                if x not in self.tracking_class.INSTANCES['active'].keys():
                    CONSOLE.log(f"No `{x}` component is in active! Please build new one with `add=['{x}']`")
                    continue               

                else:  # in active
                    del self.all_models_configs[x]['model_instance_name']

            CONSOLE.log(f"Done remove models.")



        # show info 
        CONSOLE.print(self.tracking_class.get_instances_table())


    # TODO
    def model_infer(self):
        # multi model instances, multi images, multi threads
        pass



    def requests_solver(self):
        # cope with request 

        @FLASK_APP.route(self.flask_url, methods=["POST"])
        @TIMER(prefix='post_solver function')
        def post_solver():
            # cope with POST request 

            if request.method == "POST":

                # parse input model list : add & remove
                if request.form[self.add_model_field] is not None:
                    add_list = request.form[self.add_model_field].split(',')
                    add_list = [x for x in map(lambda x: x.strip(), add_list) if len(x) != 0]

                if request.form[self.remove_model_field] is not None:
                    remove_list = request.form[self.remove_model_field].split(',')
                    remove_list = [x for x in map(lambda x: x.strip(), remove_list) if len(x) != 0]

                # build model
                self.submit(add=add_list, remove=remove_list, verbose=self.verbose)

                ys = {}  # return dict

                # images
                if request.files.get(self.image_field).filename is not None:
                    try:

                        image_input = request.files[self.image_field].read()
                        img_bytes = np.frombuffer(image_input, np.uint8)
                        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

                        # models
                        for model_name, instance in self.tracking_class.INSTANCES['active'].items():

                            yy = {}
                            ys[model_name] = []

                            # may has multi instances per conponent model
                            for hash_id, instance_info in instance.items():

                                # 
                                t = TIMER(prefix=f"{instance_info.component_name}({hash_id})")
                                # ti = INSPECTOR(prefix=f"{instance_info.component_name}({hash_id})", cpu=True, mem=True)
                                with t:
                                    # y = instance_info.weakref()([img])  # infer
                                    y = instance_info.weakref()([img, img, img])  # infer batch
                                    # print(f"> Results: {y}\n")

                                    # print(f"cpu ----> {(psutil.cpu_percent(percpu=False, interval=None))}")
                                    # print(f"mem ----> {psutil.virtual_memory().used}")


                                yy.update({str(hash_id): y})

                            # update return dict
                            ys[model_name].append(yy)


                    except Exception as E:  
                        return f"> Exception: {E}"
                else:
                    return f"No `images` key!"


            return jsonify(ys)



    def resource_info_live(self, time_last=3, refresh_time=0.5):
        with Live(self.track_resource(refresh_time=refresh_time), refresh_per_second=4, screen=False, transient=False) as live:
            t = time.time()

            while time.time() - t < time_last:
                live.update(self.track_resource(refresh_time=refresh_time))


    def inspect(self):
        CONSOLE.print(self.tracking_class.get_instances_table())





# ----------------------------------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name='all_models')
def main(cfg: DictConfig) -> None:
    # rich.print(OmegaConf.to_yaml(cfg), '\n\n')

    
    all_models_configs = get_projects_configs(configs=cfg)      # parse projects configs
    # rich.print(all_models_configs)


    all_rounder = AllRounder(all_models_configs)     # init profiler


    # all_rounder.submit(add=['playing_phone', 'mask_wearing', 'sadj'], remove=['mask_wearing', 'hhhhahahaha'], verbose=False)
    all_rounder.submit(add=['playing_phone', 'mask_wearing', 'face_mask_unknown'], remove=['face_mask_unknown'], verbose=False)
    all_rounder.inspect()
    # all_rounder.requests_solver()





if __name__ == '__main__':
    main()


