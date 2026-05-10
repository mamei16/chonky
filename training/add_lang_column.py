from pathlib import Path
import json

from datasets import load_dataset
import numpy as np

from utils import get_dataset_names, rmdir


dataset_names = get_dataset_names()

for name in dataset_names:
    ds = load_dataset("data/wikipedia_shuffled/" + "20231101." + name)
    print(name)
    ds["train"] = ds["train"].add_column(name="lang", column=[name]*len(ds["train"]))
    ds.save_to_disk(Path("data/wikipedia_shuffled_filtered/" + "20231101." + name), num_proc=16)
    rmdir(Path.home() / ".cache/huggingface/datasets" / ("20231101." + name))
    rmdir(Path("data/wikipedia_shuffled/" + "20231101." + name))
    print()
