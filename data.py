import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from typing import Iterable, List
from constants import *

def yield_tokens(data_iter:Iterable,language:str) -> List[str]:    

    for data in data_iter:
        yield token_transform[language](data[language_idx[language]])


def get_data():
    
    for lang in [SRC_lang,TGT_lang]:
        train_iter = Multi30k(split='train',language_pair=(SRC_lang,TGT_lang))

        vocab_transform[lang] = build_vocab_from_iterator(
            yield_tokens(train_iter,lang),
            min_freq=1,
            specials = special_symbols,
            special_first = True
        )

    for lang in [SRC_lang,TGT_lang]:
        vocab_transform[lang].set_default_index(unk_idx)
