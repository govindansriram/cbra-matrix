import os.path
from abc import abstractmethod

import pandas as pd
import json
import numpy as np

def base_path() -> str:
    current_file_path = __file__
    head, _ = os.path.split(current_file_path)
    head, _ = os.path.split(head)
    full_path = os.path.join(head, "benchmarks", "reports")
    return full_path

class Graph:

    _filename = ""

    class _Line:
        label: str = ""
        x_values = None
        y_values = None

        def __str__(self):
            x = f""" Line Name: {self.label}; X Values: {self.x_values}; Y Values: {self.y_values}"""

            return x

    def __init__(self):
        path = os.path.join(base_path(), self._filename)
        with open(path, "r") as f:
            report = json.load(f)

        self._df = pd.DataFrame(data=report["benchmarks"])
        self._drop_columns()

    @abstractmethod
    def _drop_columns(self):
        pass

    @abstractmethod
    def _group_by(self) -> list[pd.DataFrame]:
        pass

    @abstractmethod
    def _get_time_line(self, df) -> _Line:
        pass

    @abstractmethod
    def _get_cpu_line(self, df) -> _Line:
        pass

    @abstractmethod
    def _graph(self, lines: list[_Line]):
        pass

    def pipeline(self):
        frames = self._group_by()
        lines = list(map(self._get_time_line, frames))
        self._graph(lines)
