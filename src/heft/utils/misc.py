import numpy as np
from tqdm import tqdm
from datetime import datetime

def _timestamp() -> str:
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

def printi(msg: str):
    tqdm.write(f"{_timestamp()} \033[92m[INFO]\033[0m: {msg}")

def printw(msg: str):
    tqdm.write(f"{_timestamp()} \033[93m[WARNING]\033[0m: {msg}")

def printe(msg: str):
    tqdm.write(f"{_timestamp()} \033[91m[ERROR]\033[0m: {msg}")

def convert_np(obj):
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj