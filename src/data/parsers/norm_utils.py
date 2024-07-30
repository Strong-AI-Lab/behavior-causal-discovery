
from typing import Tuple, Any, List
from abc import ABCMeta, abstractmethod
import numpy as np


class CoordinatesNormaliser(metaclass=ABCMeta):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def setup(self, data : Any) -> None:
        raise NotImplementedError("Normaliser must implement setup method.")

    @abstractmethod
    def normalise(self, coordinates : Tuple[float, float, float]) -> Tuple[float, float, float]:
        raise NotImplementedError("Normaliser must implement normalise method.")
    
    

class MinMaxNormaliser(CoordinatesNormaliser):
    
    def __init__(self) -> None:
        super().__init__()
        self.min = None
        self.max = None

    def setup(self, data :  List[Tuple[float, float, float]]) -> None:
        data = np.array(data)
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)

    def normalise(self, coordinates : Tuple[float, float, float]) -> Tuple[float, float, float]:
        x, y, z = coordinates
        x = (x - self.min[0]) / (self.max[0] - self.min[0])
        y = (y - self.min[1]) / (self.max[1] - self.min[1])
        z = (z - self.min[2]) / (self.max[2] - self.min[2])
        return x, y, z
        

class GaussianNormaliser(CoordinatesNormaliser):
    
    def __init__(self) -> None:
        super().__init__()
        self.mean = None
        self.std = None

    def setup(self, data : List[Tuple[float, float, float]]) -> None:
        data = np.array(data)
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

    def normalise(self, coordinates : Tuple[float, float, float]) -> Tuple[float, float, float]:
        x, y, z = coordinates
        x = (x - self.mean[0]) / self.std[0]
        y = (y - self.mean[1]) / self.std[1]
        z = (z - self.mean[2]) / self.std[2]
        return x, y, z