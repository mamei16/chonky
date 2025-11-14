from pathlib import Path
import json
import lzma

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


with open("translated_most_common_last_lines.json", "r") as f:
    lang_to_last_lines = json.load(f)

with lzma.open("lsjbot_dicts.json.xz", "rb") as archive:
        json_bytes = archive.read()
lang_to_bot_title_dict = json.loads(json_bytes)


BOT_LANGUAGES = ["sv", "war", "ceb"]


def is_not_long_short(datum_dict):
    """
    Removes articles whose mean paragraph length is below 30 chars
    or whose median paragraph length is over 1500.
    """
    split_lens = [len(s) for s in datum_dict["text"].split("\n\n") if s.strip()]
    mean_len = np.mean(split_lens)
    median_len = np.median(split_lens)
    return median_len < 1500 and mean_len > 30


def more_than_two_paras(d, last_lines_dict):
    """
    Filter extremely short articles with less than 2 "real" paragraphs
    """
    splits = d["text"].split("\n\n")
    real_paragraph_count = 0
    for split in splits:
        if split.strip() in last_lines_dict:
            break
        if split.rstrip().endswith(".") or len(split) > 50:
            real_paragraph_count += 1
        if real_paragraph_count >= 2:
            break
    return real_paragraph_count >= 2


def is_not_bot_written(d, lang):
    """
    Some articles contain a lot of missing metrics and will leave an extra space in their place.
    For example: "The mountain is  km high"
    We assume that they are written by a bot or are otherwise of low quality.
    """
    if lang not in BOT_LANGUAGES:
        return True
    return "  " not in d["text"] and d["title"] not in lang_to_bot_title_dict[lang]



for name in dataset_names:
    ds = load_dataset("data/wikipedia_shuffled/" + "20231101." + name)
    print(name)
    print(len(ds["train"]))
    ds = ds.filter(lambda d: is_not_long_short(d) and more_than_two_paras(d, lang_to_last_lines[name]) and is_not_bot_written(d, name), num_proc=16)
    print(len(ds["train"]))
    ds.save_to_disk("data/wikipedia_shuffled_filtered/" + "20231101." + name, num_proc=16)
    rmdir(Path.home() / ".cache/huggingface/datasets/" / ("20231101." + name))
    rmdir(Path("data/wikipedia_shuffled/" + "20231101." + name))
    print()
