import json
from pathlib import Path

from datasets import load_dataset


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

for lang in dataset_names:
    ds = load_dataset("data/wikipedia", lang)
    shuffled_ds = ds.shuffle()
    shuffled_ds.save_to_disk(f"data/wikipedia_shuffled/{lang}", num_proc=16)
    rmdir(Path(Path.home() / f".cache/huggingface/datasets/wikipedia/{lang}"))
