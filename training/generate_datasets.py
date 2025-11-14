import os.path
from operator import itemgetter
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, IterableDataset, load_from_disk, Value, Features, List
from tqdm import tqdm
import regex
from transformers import AutoTokenizer


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


tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-multilingual-cased")


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512,
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif (
                    word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


def dataset_dict_gen(path):
    ds = load_from_disk(path)
    ids = ds.to_iterable_dataset()

    for datum_dict in tqdm(ids, total=len(ds)):
        paragraphs = regex.split("(?<!:)\n\n", datum_dict["text"])

        if len(paragraphs) < 2:
            continue

        tokens = []
        ner_tags = []

        for para in paragraphs:
            stripped_para = para.strip()
            if stripped_para in lang_to_last_lines[datum_dict["lang"]]:
                break
            if len(stripped_para) == 0:
                continue
            para_splits = para.split(" ")
            for token in para_splits:
                tokens.append(token)
                ner_tags.append(0)
            # TODO: How to deal with series of consecutive ones?
            #if ner_tags[-2] == 1:
            #    ner_tags[-2] = 0
            if len(para_splits) > 1:
                ner_tags[-1] = 1

        if not ner_tags:
            continue

        ner_tags[-1] = 0

        yield {"tokens": tokens, "ner_tags": ner_tags}


def main():
    repo_id = "mamei16/multilingual-wikipedia-paragraphs"
    dataset = Dataset.from_generator(dataset_dict_gen, gen_kwargs={"path": "data/wikipedia_shuffled_combined"},
                                     features=Features({"tokens": List(Value("string")),
                                                        "ner_tags": List(Value("int8"))}))
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, batch_size=1024, remove_columns="tokens",
                                    features=Features({"ner_tags": List(Value("bool")),
                                                       "attention_mask": List(Value("bool")),
                                                       'input_ids': List(Value('int32')),
                                                       'labels': List(Value('int8'))
                                                       }),
                                    num_proc=16)
    tokenized_dataset.push_to_hub(repo_id, "all_combined_train", num_proc=16)
    generator_cache_path = Path.home() / ".cache/huggingface/datasets/generator"
    rmdir(generator_cache_path)
    repo_id_dataset_cache_path = Path.home() / ".cache/huggingface/datasets" / repo_id.replace("/", "___")
    if os.path.exists(repo_id_dataset_cache_path):
        rmdir(repo_id_dataset_cache_path)


    for name in dataset_names:  # TODO: This leads to rate limit exceptions, so a back-off sleep logic is needed
        val_test_datasets = [
            f"data/wikipedia_shuffled/20231101.{name}/fast_val",
            f"data/wikipedia_shuffled/20231101.{name}/full_val",
            f"data/wikipedia_shuffled/20231101.{name}/test"
        ]
        for dataset_name in val_test_datasets:
            dataset = Dataset.from_generator(dataset_dict_gen, gen_kwargs={"path": dataset_name},
                                             features=Features({"tokens":List(Value("string")),
                                                                "ner_tags": List(Value("int8"))}))
            tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True,
                                            batch_size=1024, remove_columns="tokens",
                                            features=Features({"ner_tags": List(Value("bool")),
                                                               "attention_mask": List(Value("bool")),
                                                               'input_ids': List(Value('int32')),
                                                               'labels': List(Value('int8'))
                                                               }),
                                            num_proc=16)
            #tokenized_dataset.save_to_disk(f"{dataset_name}_processed")
            tokenized_dataset.push_to_hub(repo_id, name, split=dataset_name.rsplit("/", 1)[1])
            #tokenized_dataset.to_parquet(f"data/parquet_val_files/{name}/"+ dataset_name.rsplit("/", 1)[1] + "-00000-of-00001.parquet")
            if os.path.exists(generator_cache_path):
                rmdir(generator_cache_path)
            if os.path.exists(repo_id_dataset_cache_path):
                rmdir(repo_id_dataset_cache_path)


if __name__ == "__main__":
    main()
