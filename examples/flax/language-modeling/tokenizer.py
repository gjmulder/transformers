#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 17:25:09 2023

@author: mulderg
"""
import datasets

from t5_tokenizer_model import SentencePieceUnigramTokenizer
from transformers import T5Config

model_name = "flan-t5-large"

vocab_size = 32_000
input_sentence_size = None

# Initialize a dataset
dataset = datasets.load_dataset("oscar", name="unshuffled_deduplicated_no", split="train")

tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")

# Build an iterator over this dataset
def batch_iterator(input_sentence_size=None):
    if input_sentence_size is None:
        input_sentence_size = len(dataset)
    batch_length = 100
    for i in range(0, input_sentence_size, batch_length):
        yield dataset[i: i + batch_length]["text"]


# Train tokenizer
tokenizer.train_from_iterator(
    iterator=batch_iterator(input_sentence_size=input_sentence_size),
    vocab_size=vocab_size,
    show_progress=True,
)

# Save files to disk
tokenizer.save("./norwegian-%s/tokenizer.json" % model_name)

config = T5Config.from_pretrained("google/%s" % model_name, vocab_size=tokenizer.get_vocab_size())
config.save_pretrained("./norwegian-%s" % model_name)