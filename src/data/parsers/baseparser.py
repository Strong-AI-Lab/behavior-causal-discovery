
import abc
from typing import Any

class Parser(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def parse(self, data :Any, structure : Any) -> None:
        return
    
    @abc.abstractmethod
    def copy(self) -> 'Parser':
        return