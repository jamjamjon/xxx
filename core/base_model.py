import numpy as np
import onnxruntime
from pathlib import Path
import cv2
import yaml
from omegaconf import DictConfig, OmegaConf
import hydra
from abc import ABC, abstractmethod
import rich
from rich import inspect
from rich.live import Live
import sys
from flask import Flask, request, jsonify
import weakref
import inspect
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

# import torch
# import torchvision
from typing import List, Union  


# path
FILE = Path(__file__).resolve() # file resolve path
ROOT_DIR = FILE.parents[1]  # ROOT_dir
PARENT_DIR = FILE.parents[0]    # parent dir 


# add to python path
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR)) 

# modules
from core.base_predictor import Predictor 
from core.utils import (WEIGHTS_TYPE, MODEL_TYPE, IMG_FORMAT, CenterCrop, Softmax, Normalize, letterbox, 
                        parse_model_config, CONSOLE, MB, GB, KB, pysizeof, LOGGER, scale_boxes_batch, xywh2xyxy, 
                        TIMER, InstanceInfo, DeviceInfo, get_device)




# yaml path (modify this for every project)
# CONFIG_DIR = str(ROOT_DIR / "deploy/projects/playing_phone/configs")
CONFIG_DIR = str(ROOT_DIR / "deploy/projects/mask_wearing/configs")
CONFIG_NAME = "default"

# ----------------------------------------------------------------------------------------------------------------




class EnsembleModel(ABC):
    # 

    INSTANCES = {'active': {}, 'deprecated': {}}   
    # {
    #     'active': {
    #         'not_classify': {
    #             8735941799289: InstanceInfo(), ...
    #         }
    #     },
    #     'deprecated': {}
    # }


    def __init__(self, config, verbose=False, do_warmup=False):
        assert config is not None, f"{self.__class__.__name__} has no config!"
        self.verbose = verbose  # verbose 
        # CONSOLE.log(inspect.stack())

        # --------------------------------------------------------------
        # check device
        self.device: DeviceInfo = get_device(config[0].get('device', 'cpu'))  # new 
        self.test_image = config[0].get('test_image', None)
        self.description = config[0].get('description', None)
        self.model_class = config[0].get('model_class', None)
        self.component_name = config[0].get('component_name', 'untitled')   # TODO: component name
        # --------------------------------------------------------------


        # instance info
        if self.component_name not in self.__class__.INSTANCES['active'].keys():
            self.__class__.INSTANCES['active'].setdefault(self.component_name, {})
        self.__class__.INSTANCES['active'][self.component_name].update({
            hash(self): InstanceInfo(
                            weakref=weakref.ref(self),  # weakref=weakref.proxy(self),
                            uuid=uuid.uuid3(uuid.NAMESPACE_DNS, str(hash(self))).hex,
                            component_name=self.component_name,
                            ptr=hex(id(self)),
                            cur_class_name=self.__class__.__name__, 
                            base_class_names=[x.__name__ for x in self.__class__.__bases__],
                            file_name=Path(inspect.stack()[1].filename).name,    # TODO: more elegent way
                            device=self.device,
                            lineno=inspect.stack()[1].lineno,     # line number
                            date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # datetime.now().isoformat()
                        )
        })

        # build models
        self.models = {}
        if config is not None:
            for cfg in config:
                self.models.update({cfg.get('model_name'): Predictor(cfg, device=self.device, do_warmup=do_warmup)})
                # rich.print(self.models[cfg.get('model_name')].__dict__)


        # verbose
        if self.verbose:
            self._info()


    def to(self, device='cpu'):
        # device change
        # last_device = self.device  # last device 
        _new_device = get_device(device) # parse device

        if self.device == _new_device:
            LOGGER.warning(f"{hash(self)}({self.__class__.__name__}) Model already located at device: {self.device}, Nothing changed!")
        else:
            _last_device = self.device  # save last device

            # change all ensemable modules' device
            for k, v in self.models.items():   
                v._to_device(_new_device)

            # update ensemable model device info
            self.device = _new_device
            self.__class__.INSTANCES['active'][self.component_name][hash(self)].device = _new_device
            LOGGER.info(f"{hash(self)}({self.__class__.__name__}) Model changed device: {_last_device} -> {_new_device}.")

            # TODO: do warmup once when changing device
            # self.warmup(times=1, batch_size=3)   



    def __del__(self):
        # update instance deprecated
        try:

            # build component item
            if self.component_name not in self.__class__.INSTANCES['deprecated'].keys():
                self.__class__.INSTANCES['deprecated'].setdefault(self.component_name, {})

            # pop from `active` to `deprecated` parts
            self.__class__.INSTANCES['deprecated'][self.component_name].update({
                hash(self): self.__class__.INSTANCES['active'][self.component_name].pop(hash(self))
            })  

            # when component has no items, delete this component
            if len(self.__class__.INSTANCES['active'][self.component_name].keys()) == 0:
                self.__class__.INSTANCES['active'].pop(self.component_name)
            
        except Exception as E:
            CONSOLE.log(f"Exception: {E}")



    def warmup(self, times=1, batch_size=3):
        # warmup
        for k, v in self.models.items():
            v._warmup(times=times, batch_size=batch_size)
        LOGGER.info(f'> {self.__class__.__name__} class done warmup!')



    def __call__(self, x):
        # infer: need warmup at 1st time or warmup when change device!
        return self.post_process(x)

    
    # @abstractmethod
    def post_process(self, x):  # TODO: rename, logic code, must write
        # code here
        return x


    def _get_custom_methods(self):
        # get custom methods

        custom_methods_list = []
        for x in dir(self):
            if not x.startswith('__') and not x.endswith('__') and callable(eval("self." + x)):
                custom_methods_list.append(x)
        return custom_methods_list


    def _info(self):
        # model info

        for k, v in self.models.items():
            rich.inspect(v, title=k, private=False, dunder=False, all=False,)


    @property
    def ref_cnt(self):
        return sys.getrefcount(self) - 3   # # in class -> -3


    def __repr__(self):
        return self.__class__.__name__


    def __contains__(self, x):
        return x in self.__dict__


    @classmethod
    def get_instances(cls):
        return cls.INSTANCES


    @property
    def instances(self):
        return self.__class__.INSTANCES


    @property
    def base_class_names(self):
        return [x.__name__ for x in self.__class__.__bases__]


    def instances_info(self):
        CONSOLE.print(self.get_instances_table())


    @classmethod
    def get_instances_table(cls, caption='', title='Class Instances Info'):
        table = rich.table.Table(
                    title=f"\n[bold cyan]{title}", 
                    title_style='left',
                    box=rich.box.ASCII2, 
                    show_lines=False, 
                    caption=f"\n{caption}\n",
                    caption_justify='center',
                    header_style='bold cyan',
                )

        # add columns
        table.add_column("Serial Number", justify="center", style="", no_wrap=True)
        table.add_column("Ptr", justify="center", style="", no_wrap=True)
        # table.add_column("UUID", justify="center", style="", no_wrap=True)
        table.add_column("Component", justify="center", style="", no_wrap=True)
        table.add_column("Class", justify="center", style="", no_wrap=True)
        table.add_column("Base", justify="center", style="", no_wrap=True)
        table.add_column("Mem", justify="center", style="", no_wrap=True)
        # table.add_column("Ref Cnt", justify="center", style="", no_wrap=True)
        table.add_column("Device", justify="center", style="", no_wrap=True)   # new added
        table.add_column("Where", justify="center", style="")
        table.add_column("date", justify="center", style="")
        table.add_column("Status", justify="center", style="green")

        # add row
        for idx, (status, component_dict) in enumerate(cls.INSTANCES.items()):
            for idy, (component_name, instance_dict) in enumerate(component_dict.items()):
                for idz, (k, instance) in enumerate(instance_dict.items()):
                    table.add_row(
                        f"{k}",   # instance hash
                        f"{instance.ptr}",  
                        # f"{instance.uuid}",  
                        f"{component_name}",  
                        f"{instance.cur_class_name}",   
                        f"{instance.base_class_names}",   
                        f"{(pysizeof(instance.weakref()) / KB):.2f} KB" if status == 'active' else f"None",
                        # f"{instance.weakref().ref_cnt}" if status == 'active' else f"None",
                        # f"{instance.device}",   
                        f"{instance.device.type}: {instance.device.id}",   
                        f"{instance.file_name} ({instance.lineno})",   
                        f"{instance.date}",   
                        f"[green]{status}" if status == 'active' else f"[red]{status}",   
                        end_section=False
                    )

        
        return table



# --------------------------------------
#   test
# --------------------------------------

def test(cfg):
    x = EnsembleModel(cfg, verbose=True)
    # return x



# -----------------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name=CONFIG_NAME)
def main(cfg):
# def main():
    # print(OmegaConf.to_yaml(cfg))

    cfg = parse_model_config(cfg)
    # rich.print(cfg)



    ensemable_base = EnsembleModel(config=cfg, verbose=False)
    ensemable_base1 = EnsembleModel(config=cfg, verbose=False)
    ensemable_base2 = EnsembleModel(config=cfg, verbose=False)
    ensemable_base3 = EnsembleModel(config=cfg, verbose=False)
    ensemable_base4 = EnsembleModel(config=cfg, verbose=False)


    del ensemable_base3, ensemable_base4


    # CONSOLE.print(ensemable_base.get_instances_table())


    a = ensemable_base
    b = a
    c = b

    a =  ensemable_base1
    d = a


    CONSOLE.print(EnsembleModel.get_instances_table())



    # ensemable_base.info()
    # print('### ensemable_base : ', ensemable_base.get_ref_count())
    # EnsembleModel.instance_info()

    # b = ensemable_base
    # c = b
    # a = c
    # print('### ensemable_base : ', ensemable_base.get_ref_count())
    # EnsembleModel.instance_info()



    # aa = test(cfg)
    # test(cfg)
    # print('### aaa ', aa.get_ref_count())
    # print('### ensemable_base : ', ensemable_base.get_ref_count())


    # rich.print(EnsembleModel.get_instances())
    # rich.print(EnsembleModel.get_num_instances())

    # ensemble_model2 = EnsembleModel(cfg, verbose=False)
    # print('### ensemble_model2 ', ensemble_model2.get_ref_count())

    # rich.print(EnsembleModel.get_num_instances())


    # EnsembleModel().manager()



    # ensemble_model3 = EnsembleModel(cfg, verbose=False)
    # rich.print(EnsembleModel.get_instances())
    # rich.print(EnsembleModel.get_num_instances())

    # # ensemble_model3.info()
    # # print('### ', ensemble_model3.get_ref_count())


    # del ensemble_model3



    # del ensemable_base, ensemble_model3


    # ensemble_model4 = EnsembleModel(cfg, verbose=False)
    # rich.print(ensemble_model4.get_instances())
    # rich.print(EnsembleModel.get_num_instances())

    # EnsembleModel().instance_info()

    # print('### ', EnsembleModel.NUM_INSTANCES)
    # print('### ', BaseModel.NUM_INSTANCES)

    # del ensemble_model2, ensemble_model3, ensemble_model4


    # print('### ', EnsembleModel.NUM_INSTANCES)
    # print('### ', ensemble_model.get_ref_count())

    # del a, b, c

    # print('### ', ensemble_model.get_ref_count())
    # ensemble_model.info()


    # ensemble_model5 = EnsembleModel(cfg, verbose=True)


    # rich.print(EnsembleModel.get_num_instances())
    # EnsembleModel.instance_info()




if __name__ == '__main__':
    main()
