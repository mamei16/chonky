from operator import itemgetter
from collections import defaultdict
import numpy as np
import pandas as pd
from datasets import Dataset, IterableDataset, load_from_disk
from tqdm import tqdm


def dataset_dict_gen(path):
    ds = load_from_disk(path)
    ids = ds.to_iterable_dataset()

    for datum_dict in tqdm(ids, total=len(ds)): #ds["text"]:
        paragraphs = list(filter(len, datum_dict["text"].split("\n\n"))) #list(filter(len, datum_dict["text"].split(".\n\n")))

        tokens = []
        ner_tags = []

        for para in paragraphs:
            for token in para.split(" "):
                tokens.append(token)
                ner_tags.append(0)
            ner_tags[-1] = 1

        yield {"tokens": tokens, "ner_tags": ner_tags}



def main():
    raw_datasets = [
            "data/en_wikipedia_2023_train",
            "data/en_wikipedia_2023_validation",
    ]

    for dataset_name in raw_datasets:
        dataset = Dataset.from_generator(dataset_dict_gen, gen_kwargs={"path": dataset_name})
        dataset.save_to_disk(f"{dataset_name}_processed")


if __name__ == "__main__":
    main()
