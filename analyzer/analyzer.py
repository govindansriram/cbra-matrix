import os.path
import pandas as pd
import json

def base_path() -> str:
    current_file_path = __file__
    head, _ = os.path.split(current_file_path)
    head, _ = os.path.split(head)
    full_path = os.path.join(head, "benchmarks", "reports")
    return full_path

def read_benchmark():
    full_path = os.path.join(base_path(), "benchmark_matrix.json")

    with open(full_path, "r") as f:
        report = json.load(f)

    df = pd.DataFrame(data=report["benchmarks"])
    df.drop(columns=[
        "name",
        "run_name",
        "family_index",
        "per_family_instance_index",
        "repetitions",
        "repetition_index",
        "threads",
        "run_type"
    ], inplace=True)

    print(df)

if __name__ == '__main__':
    read_benchmark()