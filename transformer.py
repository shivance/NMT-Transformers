import torch
import torch.nn as nn
import math
from constants import *
from torch import Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class PositionalEncoding(nn.Module):

    def __init__(self,emb_size,dropout,maxlen=5000):
        super(PositionalEncoding,self).__init__()
        #torch.arange returns 1D tensor
        den  = torch.exp(-torch.arange(0,emb_size,2)*math.log(1000)/emb_size)
        pos = torch.arange(0,maxlen).reshape(maxlen,1)
        pos_embedding = torch.zeros(maxlen,emb_size)
        pos_embedding[:,0::2] = torch.sin(pos*den)
        pos_embedding[:,1::2] = torch.cos(pos*den)
        
        print("Shape of pos embedding = ",pos_embedding.shape,end=" ")
        pos_embedding = pos_embedding.unsqueeze(-2)
        print("Shape of pos embedding = ",pos_embedding.shape,end=" ")

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding',pos_embedding)

    def forward(self,token_embedding):
        return self.dropout(token_embedding+self.pos_embedding[:token_embedding.size(0),:])


class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size,emb_size):
        super(TokenEmbedding,self).__init__()
        self.embedding = nn.Embedding(vocab_size,emb_size)
        self.emb_size = emb_size

    def forward(self,tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class S2S_Transformer(nn.Module):

    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        emb_size,
        nhead,
        src_vocab_size,
        tgt_vocab_size,
        dim_feedforward=512,
        dropout=0.1
    ) -> None:
        
        super(S2S_Transformer,self).__init__()

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size,emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size,emb_size)
        self.positional_encoding = PositionalEncoding(emb_size,dropout=dropout)

    def forward(
        self,
        src:Tensor,
        tgt:Tensor,
        src_mask:Tensor,
        tgt_mask:Tensor,
        src_padding_mask:Tensor,
        tgt_padding_mask:Tensor,
        memory_key_padding_mask:Tensor
        ):

        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        out = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask
        )

        return self.generator(out)

    def encode(self,src,src_mask):
   
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)),
            src_mask
        )
    
    def decode(self,tgt,memory,tgt_mask):
        
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory,
            tgt_mask
        )



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


def sequential_transforms(*transforms):

    def func(txt_input):
        for tfm in transforms:
            txt_input = tfm(txt_input)
        return txt_input
    
    return func

def tensor_transform(token_ids:List[int]):
    return torch.cat((torch.tensor([bos_idx]),torch.tensor(token_ids),torch.tensor([eos_idx])))


#function to generate output sequence

def decode_greedy(model,src,src_mask,max_len,start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src,src_mask)
    y = torch.ones(1,1).fill_(start_symbol).type(torch.long).to(device)
    
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(y.size(0))).type(torch.bool).to(device)
        
        out = model.decode(y,memory,tgt_mask).transpose(0,1)
        prob = model.generator(out[:,-1])
        _, next_word = torch.max(prob,dim=-1)
        next_word = next_word.item()

        y = torch.cat([y,torch.ones(1,1).type_as(src.data).fill_(next_word)],dim=0)

        if next_word == eos_idx:
            break

    return y

text_transform = {}
for lang in [SRC_lang,TGT_lang]:
    text_transform[lang] = sequential_transforms(
                                token_transform[lang],
                                vocab_transform[lang],
                                tensor_transform)


def translate(model:nn.Module,src_sentence:str):
    model.eval()
    src = text_transform[SRC_lang](src_sentence).view(-1,1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens,num_tokens)).type(torch.bool)
    
    tgt_tokens = decode_greedy(
        model,
        src,
        src_mask,
        max_len=num_tokens+5,
        start_symbol=bos_idx
    ).flatten()


    return " ".join(
        vocab_transform[TGT_lang].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>","").replace("<eos>","")
