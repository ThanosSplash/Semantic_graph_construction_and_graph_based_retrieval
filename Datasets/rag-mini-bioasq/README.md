---
language:
- en
license: cc-by-2.5
task_categories:
- question-answering
- sentence-similarity
dataset_info:
- config_name: question-answer-passages
  features:
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: id
    dtype: int64
  - name: relevant_passage_ids
    sequence: int64
  splits:
  - name: train
    num_bytes: 1615888.0491629583
    num_examples: 4012
  - name: test
    num_bytes: 284753.9508370418
    num_examples: 707
  download_size: 1309572
  dataset_size: 1900642.0
- config_name: text-corpus
  features:
  - name: passage
    dtype: string
  - name: id
    dtype: int64
  splits:
  - name: test
    num_bytes: 60166919
    num_examples: 40181
  download_size: 35304894
  dataset_size: 60166919
configs:
- config_name: question-answer-passages
  data_files:
  - split: train
    path: question-answer-passages/train-*
  - split: test
    path: question-answer-passages/test-*
- config_name: text-corpus
  data_files:
  - split: test
    path: text-corpus/test-*
tags:
- biology
- medical
- rag
---
This dataset is a subset of a training dataset by [the BioASQ Challenge](http://www.bioasq.org/), which is available [here](http://participants-area.bioasq.org/Tasks/11b/trainingDataset/).

It is derived from [`rag-datasets/rag-mini-bioasq`](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq).

Modifications include:
- filling in missing passages (some of them contained `"nan"` instead of actual text),
- changing `relevant_passage_ids`' type from string to sequence of ints,
- deduplicating the passages (removed 40 duplicates) and fixing the `relevant_passage_ids` in QAP triplets to point to the corrected, deduplicated passages' ids,
- splitting QAP triplets into train and test splits.