import torch
torch.cuda.empty_cache()

from data import *
from constants import *
from transformer import *
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from timeit import default_timer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



transformer = S2S_Transformer(
        NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS,
        EMB_SIZE,
        NHEAD,
        SRC_VOCAB_SIZE,
        TGT_VOCAB_SIZE,
        FFN_HID_DIM    
    )

if os.path.exists(MODEL_PATH):
    transformer.load_state_dict(torch.load(MODEL_PATH))
else:
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
            )



# combine (collate) data into  batch of tensors                     
def collate_fn(batch):
    src_batch, tgt_batch = [], []

    for src_sample,tgt_sample in batch:
        src_batch.append(text_transform[SRC_lang](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_lang](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch,padding_value = pad_idx)
    tgt_batch = pad_sequence(tgt_batch,padding_value = pad_idx)

    return src_batch, tgt_batch



def train_epoch(model,optimizer):
    model.train()
    running_loss = 0

    train_iter = Multi30k(split="train",language_pair=(SRC_lang,TGT_lang))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE,collate_fn=collate_fn)

    for src,tgt in train_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_in = tgt[:-1,:]
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src,tgt_in)

        logits = model(
                src,
                tgt_in,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask
            )

        optimizer.zero_grad()
        
        tgt_out = tgt[1:,:]
        loss = loss_fn(logits.reshape(-1,logits.shape[-1]),tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

    return running_loss/len(train_dataloader)


def evaluate(model):
    model.eval()

    running_loss = 0
    val_iter = Multi30k(split="valid",language_pair=(SRC_lang,TGT_lang))
    val_dataloader = DataLoader(val_iter,batch_size = BATCH_SIZE, collate_fn=collate_fn)

    for src,tgt in val_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_in = tgt[:-1,:]
        src_mask,tgt_mask,src_padding_mask,tgt_padding_mask = create_mask(src,tgt_in)

        logits = model(src,tgt_in,src_mask,tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        
        running_loss+=loss.item()

    
    return running_loss/len(val_dataloader)

print("Starting Training")

for epoch in range(NUM_EPOCHS):
    st = default_timer()
    train_loss = train_epoch(transformer,optimizer)
    ft = default_timer()

    val_loss = evaluate(transformer)
    print(f"Epoch : {epoch}  Train Loss : {train_loss:.3f}  Val Loss : {val_loss:.3f}")
    print(f"Epoch Time : {(ft-st):.3f}s")

torch.save(transformer.state_dict(),MODEL_PATH)


