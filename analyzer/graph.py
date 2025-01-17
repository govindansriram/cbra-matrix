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


def parse_frame(json_path: str) -> pd.DataFrame:
    with open(json_path, "r") as f:
        report = json.load(f)

    return pd.DataFrame(data=report["benchmarks"])


def read_txt_file(text_path) -> tuple[str, int]:
    with open(text_path, "r") as f:
        line = f.readline()
        line2 = f.readline()

    flags = line
    threads = int(line2)

    return flags, threads



class Graph:

    _benchmark_type = ""

    class _Line:
        label: str = ""
        x_values = None
        y_values = None

        def __str__(self):
            x = f""" Line Name: {self.label}; X Values: {self.x_values}; Y Values: {self.y_values}"""

            return x

    def __init__(self, benchmark_names: list[str]):

        self._df = None
        first = True

        for name in benchmark_names:
            path = os.path.join(base_path(), self._benchmark_type, name)
            json_path = os.path.join(path, "report.json")
            text_path = os.path.join(path, "config.txt")

            flags, threads = read_txt_file(text_path)

            settings = f"FLAGS: {flags} | THREADS: {threads}"
            frame = parse_frame(json_path)
            frame["settings"] = settings
            self._drop_columns(frame)

            if first:
                self._df = frame
                first = False
                continue

            self._df = pd.concat([self._df, frame])

    @abstractmethod
    def _drop_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
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
