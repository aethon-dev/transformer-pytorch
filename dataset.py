import torch
import torch.nn as nn
from torch.utils.data import Dataset

from tokenizers import Tokenizer

class BilingualDataset(Dataset):
  def __init__(self, ds, tokenizer_src: Tokenizer, tokeinzer_tgt: Tokenizer, lang_src: str, lang_tgt: str, seq_len: int) -> None:
    super().__init__()
    self.ds = ds
    self.tokenizer_src = tokenizer_src
    self.tokenizer_tgt = tokeinzer_tgt
    self.lang_src = lang_src
    self.lang_tgt = lang_tgt
    self.seq_len = seq_len

    self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
    self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
    self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

  def __len__(self):
    return len(self.ds)
    
  def __getitem__(self, index):
    src_tgt_pair = self.ds[index]
    src_text = src_tgt_pair['translation'][self.lang_src]
    tgt_text = src_tgt_pair['translation'][self.lang_tgt]

    enc_input_tokens = self.tokenizer_src.encode(src_text).ids
    dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

    enc_padding_len = self.seq_len - len(enc_input_tokens) - 2 # Reduce 2 for [sos] and [eos]
    dec_padding_len = self.seq_len - len(dec_input_tokens) - 1 # For the decoder input we only add [sos], so reduce 1

    if enc_padding_len < 0 or dec_padding_len < 0:
      raise ValueError('Sentence is too long')
    
    encoder_input = torch.cat(
      [
        self.sos_token,
        torch.tensor(enc_input_tokens, dtype=torch.int64),
        self.eos_token,
        torch.tensor([self.pad_token]*enc_padding_len, dtype=torch.int64)
      ]
    )

    decoder_input = torch.cat(
      [
        self.sos_token,
        torch.tensor(dec_input_tokens, dtype=torch.int64),
        torch.tensor([self.pad_token]*dec_padding_len, dtype=torch.int64)
      ]
    )

    label = torch.cat(
      [
        torch.tensor(dec_input_tokens, dtype=torch.int64),
        self.eos_token,
        torch.tensor([self.pad_token]*dec_padding_len, dtype=torch.int64)
      ]
    )

    assert encoder_input.size(0) == self.seq_len
    assert decoder_input.size(0) == self.seq_len
    assert label.size(0) == self.seq_len

    return {
      'encoder_input': encoder_input,
      'decoder_input': decoder_input,
      'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1 x 1 x seq_len)
      'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
      'label': label,
      'src_text': src_text,
      'tgt_text': tgt_text
    }
  
def causal_mask(size: int):
  mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
  return mask == 0