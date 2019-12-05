# Subword-alignment based cross-lingual word embeddings

This is the official implementation of our paper `Subword-alignment based cross-lingual word embeddings` in CICLing 2019, which learns cross-lingual word embeddings by exploiting subword-alignment

## Setup

Clone this repository

```
$ git clone https://github.com/jyori112/sabclwe
```

Then install required python packages
```
$ pip -r install requirements.txt
```

### Other Requirements

1. mpaligner

Please install `mpaligner` into `mpaligner_0.97` (alignment tool to obtain subword-alignment model) from: https://osdn.net/projects/mpaligner/.

2. VecMap

Please install `vecmap` (tool to obtain cross-lingual word embeddings) from: https://github.com/artetxem/vecmap.

## Usage

For the exact usage of `mpaligner` and `vecmap`, please read the official documentations.

### Dictionary induction

Given cross-lingual (bilingual) word embeddings `[LANG1_EMB]` and `[LANG2_EMB]`, first, induce bilingual dictionary by

```
$ python -m induce_dict [LANG1_EMB] [LANG2_EMB] --csls 10 > [DICT_PATH]
```
The resulting file (`[DICT_PATH]`) contains two words followed by the similarity.

### Align dictionary

To train an align model, we first preprocess the dictionary file

```
$ cat [DICT_PATH]| cut -f1,2| perl mpaligner_0.97/script/separate_for_char.pl > [PREPROCESSED_DICT_PATH]
```

Then, we train the alignment model

```
$ mpaligner_0.97/mpaligner -i [PREPROCESSED_DICT_PATH] -s >
```
This will produce alignment file in `[PREPROCESSED_DICT_PATH].align`

Finally, we reformat the resulting file by
```
$ python -m parse_aligned < [PREPROCESSED_DICT_PATH].align| sort -rnk3 > [ALIGNMENT_DICT_PATH]
```

### Filter dictionary

To filter dictionary by `[THRESHOLD]`
```
$ awk '{ if ($3 > [THRESHOLD]) print $1,$2}' < [ALIGNMENT_DICT_PATH] > [FILTERED_DICT]
```

### Train cross-lingual word embeddings from the filtered dictionary

Use `vecmap` tool to obtain cross-lingual word embeddings
```
$ python vecmap/map_embeddings.py [LANG1_EMB] [LANG2_EMB] [LANG1_EMB_OUTPUT] [LANG2_EMB_OUTPUT] --supervised
```

## Citation

To Appear.