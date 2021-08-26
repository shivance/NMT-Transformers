import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k


SRC_lang = 'en'
TGT_lang = 'de'

token_transform,vocab_transform = {}, {}

token_transform[SRC_lang] = get_tokenizer('spacy',language='en_core_web_sm')
token_transform[TGT_lang] = get_tokenizer('spacy',language='de_core_news_sm')

#yield list of token
language_idx = {SRC_lang:1,TGT_lang:0}

unk_idx, pad_idx, bos_idx, eos_idx = 0,1,2,3
special_symbols = ['<unk>','<pad>','<bos>','<eos>']


SRC_VOCAB_SIZE = len(vocab_transform[SRC_lang])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_lang])
EMB_SIZE = 512
NHEAD = 0
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

