import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from data import *


get_data()

SRC_VOCAB_SIZE = len(vocab_transform[SRC_lang])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_lang])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NUM_EPOCHS = 3
MODEL_PATH = "./saved_model/nmt_transformer.pth"
