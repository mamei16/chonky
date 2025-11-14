import os
from pathlib import Path
import json
import uuid

from datasets import load_dataset, interleave_datasets, load_from_disk, Dataset
import numpy as np
from tqdm import tqdm



class _DatasetGeneratorPickleHack:
    def __init__(self, generator, generator_id=None):
        self.generator = generator
        self.generator_id = (
            generator_id if generator_id is not None else str(uuid.uuid4())
        )

    def __call__(self, *args, **kwargs):
        return self.generator(*kwargs, **kwargs)

    def __reduce__(self):
        return (_DatasetGeneratorPickleHack_raise, (self.generator_id,))


def _DatasetGeneratorPickleHack_raise(*args, **kwargs):
    raise AssertionError("cannot actually unpickle _DatasetGeneratorPickleHack!")


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


code_to_lang = {}
with open("wikipedia_language_codes.txt", "r") as f:
    s = f.read()
splits = s.split("\n\n\n")
for split in splits:
    sub_split = split.strip().split("\n")
    code_to_lang[sub_split[1]] = sub_split[0]


lang_to_iter_ds = {}
lang_to_idx = {}
lang_counts = []
ALPHA = 0.5

lang_to_final_idx = {}

def interleave_dict_gen(_lang_to_iter_ds, _lang_to_idx, ds_names, _lang_counts):
    rng = np.random.default_rng()
    i = 0
    total_count = sum(_lang_counts)
    while len(ds_names) > 0:
        prob_list = [(count / total_count)**ALPHA for count in _lang_counts]
        summed_probs = sum(prob_list)
        prob_list = [prob / summed_probs for prob in prob_list]
        i += 1
        chosen_lang = rng.choice(ds_names, p=prob_list, shuffle=False)

        chosen_lang_idx = _lang_to_idx[chosen_lang]
        _lang_counts[chosen_lang_idx] -= 1
        if _lang_counts[chosen_lang_idx] == 0:
            print()
            print(code_to_lang[chosen_lang], i)
            lang_to_final_idx[code_to_lang[chosen_lang]] = i
            _lang_counts.pop(chosen_lang_idx)
            ds_names.pop(chosen_lang_idx)
            _lang_to_idx = {lang: i for i, lang in enumerate(ds_names)}
        total_count -= 1
        yield next(lang_to_iter_ds[chosen_lang].generator)


for i, name in enumerate(dataset_names):
    language_full_name = code_to_lang[name]
    print(name, language_full_name)

    ds = load_from_disk(f"data/wikipedia_shuffled/20231101.{name}/train")
    lang_counts.append(len(ds))
    lang_to_idx[name] = i

    lang_to_iter_ds[name] = _DatasetGeneratorPickleHack(iter(ds))

dataset = Dataset.from_generator(interleave_dict_gen, gen_kwargs={"_lang_to_iter_ds": lang_to_iter_ds,
                                                                  "_lang_to_idx": lang_to_idx,
                                                                  "ds_names": dataset_names, "_lang_counts": lang_counts})
dataset.save_to_disk(f"data/wikipedia_shuffled_combined", num_proc=16)
rmdir(Path.home() / "/.cache/huggingface/datasets/generator")

with open("lang_to_final_idx.json", "w") as f:
    json.dump(lang_to_final_idx, f)
