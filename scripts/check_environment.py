"""Print the software and GPU details needed to reproduce a run."""

import platform
import sys

import numpy
import pandas
import sklearn
import torch
import transformers
import yaml


def main():
    """Print one simple environment report to standard output."""
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"NumPy: {numpy.__version__}")
    print(f"pandas: {pandas.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"PyYAML: {yaml.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA: {torch.version.cuda}")

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            memory_gb = properties.total_memory / (1024 ** 3)
            print(f"GPU {index}: {properties.name}")
            print(f"GPU {index} memory: {memory_gb:.2f} GB")
    else:
        print("GPU: none available to PyTorch")


if __name__ == "__main__":
    main()

