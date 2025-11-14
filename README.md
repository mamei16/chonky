# Chonky

__Chonky__ is a Python library that intelligently segments text into meaningful semantic chunks using a fine-tuned transformer model. This library can be used in the RAG systems.
## Difference Between This Fork and Original Repo
This fork provides optimized inference and improved post-processing.

## Installation
To install the package from this fork:
```
pip install git+https://github.com/mamei16/chonky
```

## Usage:

```python
from chonky import ParagraphSplitter

# on the first run it will download the transformer model
splitter = ParagraphSplitter(device="cpu")

# Or you can select the model
# splitter = ParagraphSplitter(
#  model_id="mirth/chonky_modernbert_base_1",
#  device="cpu"
# )

text = """Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep. The first programs I tried writing were on the IBM 1401 that our school district used for what was then called "data processing." This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the basement of our junior high school, and my friend Rich Draves and I got permission to use it. It was like a mini Bond villain's lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — sitting up on a raised floor under bright fluorescent lights."""

for chunk in splitter(text):
  print(chunk)
  print("--")
```

### Sample Output
Using [mamei16/chonky_distilbert-base-multilingual-cased](mamei16/chonky_distilbert-base-multilingual-cased)
```
Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.
--
The first programs I tried writing were on the IBM 1401 that our school district used for what was then called "data processing." This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the basement of our junior high school, and my friend Rich Draves and I got permission to use it. It was like a mini Bond villain's lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — sitting up on a raised floor under bright fluorescent lights.
--
```


## Supported models

| Model ID                                                                                                 | Seq Length | Number of Params |
| -------------------------------------------------------------------------------------------------------- | ---------- | ---------------- |
| [mamei16/chonky_distilbert-base-multilingual-cased](mamei16/chonky_distilbert-base-multilingual-cased) | 512 | 134M |
| [mirth/chonky_mmbert_small_multilingual_1](https://huggingface.co/mirth/chonky_mmbert_small_multilingual_1) | 1024       | 140M             |
| [mamei16/chonky_mdistilbert-base-english-cased](https://huggingface.co/mamei16/chonky_mdistilbert-base-english-cased) | 512 | 134M |
| [mamei16/chonky_distilbert_base_uncased_1.1](https://huggingface.co/mamei16/chonky_distilbert_base_uncased_1.1) | 512 | 67M |
| [mirth/chonky_modernbert_large_1](https://huggingface.co/mirth/chonky_modernbert_large_1)                | 1024       | 396M             |
| [mirth/chonky_modernbert_base_1](https://huggingface.co/mirth/chonky_modernbert_base_1)                  | 1024       | 150M             |
| [mirth/chonky_distilbert_base_uncased_1](https://huggingface.co/mirth/chonky_distilbert_base_uncased_1)  | 512        | 66.4M            |