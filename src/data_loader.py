from pathlib import Path
import pandas as pd


def load_gait_signal(file_path):
    """
    Load one gait signal file and assign column names.
    """
    df = pd.read_csv(file_path, sep=r"\s+", header=None)

    columns = (
        ["time"]
        + [f"left_sensor_{i}" for i in range(1, 9)]
        + [f"right_sensor_{i}" for i in range(1, 9)]
        + ["left_total_force", "right_total_force"]
    )

    if df.shape[1] != len(columns):
        raise ValueError(
            f"{Path(file_path).name} has {df.shape[1]} columns, "
            f"but expected {len(columns)}"
        )

    df.columns = columns
    return df