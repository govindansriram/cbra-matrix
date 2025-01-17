from abc import ABC
import pandas as pd
import matplotlib.pyplot as plt
from graph import Graph


class DotProductGraph(Graph, ABC):
    _benchmark_type = "benchmark_matrix"

    def __init__(self, benchmark_names: list[str]):
        super().__init__(benchmark_names)

    def _drop_columns(self, frame: pd.DataFrame):
        frame.drop(columns=[
            "name",
            "run_name",
            "family_index",
            "per_family_instance_index",
            "repetitions",
            "repetition_index",
            "threads",
            "run_type"
        ], inplace=True)

    @staticmethod
    def _clean_frame(df: pd.DataFrame):
        df.sort_values(by=['columns'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["shape"] = df["columns"].astype('int')
        df["type"] = df["type"].astype('int')
        df.drop(columns=["rows", "columns"], inplace=True)
        return df

    @staticmethod
    def _number_to_label(num: int) -> str:
        match num:
            case 0:
                return "naive"
            case 1:
                return "parallel"
            case 2:
                return "parallel blocked"

    def _get_cpu_line(self, df) -> Graph._Line:
        line = Graph._Line()
        line.label = DotProductGraph._number_to_label(df["type"])
        line.x_values = df["shape"].to_numpy()
        line.y_values = df["cpu_time"].to_numpy()

        return line

    def _get_time_line(self, df) -> Graph._Line:
        line = Graph._Line()
        line.label = DotProductGraph._number_to_label(df["type"][0]) + " " + df["settings"][0]
        line.x_values = df["shape"].to_numpy()
        line.y_values = df["real_time"].to_numpy()

        return line

    def _group_by(self) -> list[pd.DataFrame]:
        group = self._df.groupby(['type', 'settings'])
        frame_list = [frame for _, frame in group]
        return list(map(DotProductGraph._clean_frame, frame_list))

    def _graph(self, lines: list[Graph._Line]):
        fig, ax = plt.subplots()

        for line in lines:
            ax.plot(line.x_values, line.y_values, label=line.label)

        ax.set_title("batched dot product computation time")
        ax.set_xlabel('matrix size')
        ax.set_ylabel('nanoseconds')

        ax.legend()
        plt.show()

