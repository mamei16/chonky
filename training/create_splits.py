import os
from pathlib import Path
import json

from datasets import load_dataset, concatenate_datasets, load_from_disk
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


lang_to_val_sizes = {}
with open("language_groups.json", "r") as f:
    lang_group_dicts = json.load(f)["Clusters"]

for d in lang_group_dicts:
    for lang in d["Languages"]:
        lang_to_val_sizes[lang] = {"fast": d["FastEvalPerLanguage"],
                                   "full": d["FullEvalPerLanguage"]}


code_to_lang = {}
with open("wikipedia_language_codes.txt", "r") as f:
    s = f.read()
splits = s.split("\n\n\n")
for split in splits:
    sub_split = split.strip().split("\n")
    code_to_lang[sub_split[1]] = sub_split[0]


fast_val_datasets = []
full_val_datasets = []
test_datasets = []

for name in dataset_names:
    if name == "bn":
        continue
    language_full_name = code_to_lang[name]
    print(name, language_full_name)

    ds = load_from_disk("data/wikipedia_shuffled/" + "20231101." + name)
    full_len = len(ds["train"])

    if (full_len * 0.1) > 10000:
        test_size = 10000
    else:
        test_size = int(full_len * 0.1)

    ds_test = ds["train"].take(test_size)

    language_full_name = code_to_lang[name]

    fast_size = lang_to_val_sizes[language_full_name]["fast"]
    full_val_size = lang_to_val_sizes[language_full_name]["full"]
    fast_full_ratio = fast_size / full_val_size

    if full_val_size > int(full_len * 0.1):
        full_val_size = int(full_len * 0.1)
        fast_size = int(fast_full_ratio * full_val_size)

    ds_full_val = ds["train"].select(range(test_size, test_size+full_val_size))
    ds_fast_val = ds_full_val.take(fast_size)

    ds_train = ds["train"].select(range(test_size+full_val_size, full_len))

    assert len(ds_train) + len(ds_test) + len(ds_full_val) == full_len, print(len(ds_train), len(ds_test), len(ds_full_val) )

    ds_train.save_to_disk(f"data/wikipedia_shuffled_split/20231101.{name}/train")
    ds_test.save_to_disk(f"data/wikipedia_shuffled_split/20231101.{name}/test")

    ds_fast_val.save_to_disk(f"data/wikipedia_shuffled_split/20231101.{name}/fast_val")
    ds_full_val.save_to_disk(f"data/wikipedia_shuffled_split/20231101.{name}/full_val")

    ds_cache_dir = Path.home() / "/.cache/huggingface/datasets" / ("20231101." + name)
    if os.path.exists(ds_cache_dir):
        rmdir(ds_cache_dir)
    rmdir(Path(f"data/wikipedia_shuffled/20231101.{name}/train"))
    print()
