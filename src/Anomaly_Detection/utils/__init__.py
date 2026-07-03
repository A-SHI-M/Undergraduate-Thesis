import os
from .visualizations import Visualizer


def create_directories(paths: list):
    for path in paths:
        os.makedirs(path, exist_ok=True)


__all__ = ["create_directories", "Visualizer"]
