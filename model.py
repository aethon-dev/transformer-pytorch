import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
  def __init__(self, d_model: int, vocab_size: int) -> None:
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    return self.embedding * math.sqrt(self.d_model)
  

class PositionalEmbedding(nn.Module):
  def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)

    # Create a matrix of shape (seq_len x d_model) to hold the positional embeddings
    pe = torch.zeros(seq_len, d_model)

    # Create a column vector of shape (seq_len x 1) of indices for each position 
    # in the input sequence (a column matrix really...)
    positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

    # In the paper, the position for each index is calculated as follows
    # PE(pos, 2i)   = sin(pos/(10000 ** (2i/d_model))) 
    # PE(pos, 2i+1) = cos(pos/(10000 ** (2i/d_model)))
    # Here we take a slightly different, but numerically more stable approach
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    # Apply sin and cos functions to alternating positions
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)

    # Add batch dimention to the encodings to make it of shape (1 x seq_len x d_model)
    # This way it will broadcast to the entire batch of x
    pe = pe.unsqueeze(0)

    # Add the encodings to the buffer so it will be serialized to the disk
    # along with other model parameters
    self.register_buffer('pe', pe)

  def forward(self, x):
    # x will be of shape (batch_size x seq_len x d_model)
    # So we pick the positional embeddings up to the current
    # input sequence's length by indexing [:x.shape[1]] and
    # add to it. (x.shape[1] will have the length of the current
    # input sequence)
    x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
    return self.dropout(x)
