import json
from pathlib import Path

import regex


SPLIT_PATTERN = regex.compile(r"(?<!:|：|﹕|︰)\n\n")

ENDS_WITH_PERIOD_PATTERN = regex.compile(
    r".*(\."   # Latin/Cyrillic/Greek/Armenian: period + space + uppercase
    r"|۔"   # Arabic script has no case, but if mixed with Latin, allow uppercase
    r"|।"               # Devanagari danda
    r"|॥"               # Devanagari double danda
    r"|。"               # Chinese/Japanese full stop
    r"|።"                # Ethiopic
    r"|།"               # Tibetan shad
    r"|༎"               # Tibetan double shad
    r"|။"               # Burmese
    r"|។"               # Khmer
    r"|ໆ"               # Lao
    r"|᠃"               # Mongolian
    r"|෴)$"               # Sinhala kundaliya
)


def get_dataset_names():
    with open("langs_to_keep.txt", "r") as f:
        return list(filter(len, f.read().split('\n')))


def get_lang_to_last_line():
    with open("translated_most_common_last_lines.json", "r") as f:
        return json.load(f)


def rmdir(directory):
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


