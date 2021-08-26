from data import *
import torch
from constants import *
from transformer import *
import torch.nn as nn

device = ("cuda" if torch.cuda.is_available() else "cpu")

def generate_square_subsequent_mask(sz):
    mask = (
        torch.triu(
            torch.ones(
                (sz,sz),
                device = device
            )
        )==1
    ).transpose(0,1)

    mask = mask.float().masked_fill(mask==0,float('-inf')).masked_fill(mask==1,float(0.0))

    return mask

#During training, we need a subsequent word mask that will 
#prevent model to look into the future words when making predictions

def create_mask(src,tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len,src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0,1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0,1)

    return src_mask,tgt_mask,src_padding_mask,tgt_padding_mask

transformer = S2S_Transformer(
        NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS,
        EMB_SIZE,
        NHEAD,
        SRC_VOCAB_SIZE,
        TGT_VOCAB_SIZE,
        FFN_HID_DIM    
    )

for param in transformer.parameters():
    if param.dim()>1:
        nn.init.xavier_uniform_(param)

transformer = transformer.to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index = pad_idx)

optimizer = torch.optim.Adam(
                transformer.parameters(),
                lr=0.0001,
                betas=(0.9,0.98),
                eps=1e-9,
                momentum=0.9
            )

