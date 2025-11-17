# visualization/plot_base.py
from abc import ABC, abstractmethod

from typing import Any,Callable, Dict, Type, dict
import numpy as np


STORAGE_REGISTRY: Dict[str, Type["Storage"]] = {}



class Storage(ABC):
    def __init__(self, args: dict):
        super().__init__()
        self.args = args
    
    @abstractmethod
    def store(data: any) -> bool:
        pass
