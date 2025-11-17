from lfp_analysis.registry import register
from lfp_analysis.registry.base import REGISTRIES

from .base import Storage
from scipy.io import savemat


import numpy as np

@register("storages","npy")
class Numpysaver(Storage):
    def __init__(self, args: dict):
        super().__init__(args)
    def store(self,data: np.ndarray):
        np.save(self.args["path"],data)

@register("storages","npz")
class MultipleNumpysaver(Storage):
    def __init__(self, args: dict):
        super().__init__(args)
    def store(self,data: list):
        for i in data:
            self.save(self.args[i["id"]]["path"],i["data"])
    def save(self,path: str, data: np.ndarray):
        pass


@register("storages","mat")
class Matsaver(Storage):
    def __init__(self, args: dict):
        super().__init__(args)
    def store(self,data: list):
        for i in data:
            self.save(self.args[i["id"]]["path"],i["data"],self.args[i["id"]]["variable_name"])
    def save(self,path: str, data: np.ndarray, variable: str):
        savemat(path, {variable: data})