from pathlib import Path
import json

from datasets import load_dataset
import numpy as np



def rmdir(directory):
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


with open("langs_to_keep.txt", "r") as f:
    dataset_names = list(filter(len, f.read().split('\n')))



for name in dataset_names:
    ds = load_dataset("data/wikipedia_shuffled/" + "20231101." + name)
    print(name)
    ds["train"] = ds["train"].add_column(name="lang", column=[name]*len(ds["train"]))
    ds.save_to_disk("data/wikipedia_shuffled_filtered/" + "20231101." + name, num_proc=16)
    rmdir(Path.home() / ".cache/huggingface/datasets" / ("20231101." + name))
    rmdir(Path("data/wikipedia_shuffled/" + "20231101." + name))
    print()
