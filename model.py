import torch
import torch.nn as nn
import torch.nn.functional as F
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

    # Add batch dimension to the encodings to make it of shape (1 x seq_len x d_model)
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
  

# Here we could have used nn.LayerNorm class but its constructor
# requires the normalization dimension sizes at instantiation.
# As this layer can be a part of any block, we cannot know that 
# in advance, so we calculate the layer norm on the fly.
# Not using nn.functional.layer_norm either until I figure out
# how it's affecting the computational graph
class LayerNorm(nn.Module):
  def __init__(self, eps: float = 1e-5) -> None:
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(1))
    self.bias = nn.Parameter(torch.zeros(1))

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (self.alpha * (x - mean) / (std + self.eps)) + self.bias


class FeedForward(nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
    super().__init__()
    self.linear1 = nn.Linear(d_model, d_ff)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(d_ff, d_model)

  def forward(self, x):
    return self.linear2(self.dropout(torch.relu(self.linear1(x))))
  

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int, h: int, dropout: float) -> None:
    super().__init__()
    assert d_model % h == 0, "d_model is not divisible by h"

    self.d_model = d_model
    self.h = h
    self.d_k = d_model // h

    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_o = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)

  @staticmethod
  def attention(query, key, value, mask, d_k: int, dropout: nn.Dropout):
    # This produces (batch x h x seq_len x seq_len)
    attn_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
      attn_scores.masked_fill_(mask == 0, -1e9)

    attn_scores = attn_scores.softmax(dim=-1)

    if dropout is not None:
      attn_scores = dropout(attn_scores)

    return (attn_scores @ value), attn_scores



  def forward(self, q, k, v, mask):
    # Break the embedding dimension into h number of heads and reorder the matrices
    # so that the for each head there is a (seq_len x d_k) tensor
    # (batch x seq_len x d_model) --> (batch x seq_len x d_model) --> (batch x seq_len x h x d_k)
    # --> (batch x h x seq_len x d_k)
    query = self.w_q(q).view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2)
    key = self.w_k(k).view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)
    value = self.w_v(v).view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2)

    x, self.attn_scores = MultiHeadAttention.attention(query, key, value, mask, self.d_k, self.dropout)

    # Reorganize output to the shape (batch x seq_len x d_model)
    # We use x.contiguous() to make sure the entire x is in contiguous memory
    x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

    return self.w_o(x)
  

class ResidualConnection(nn.Module):
  def __init__(self, dropout: float) -> None:
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNorm()

  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x)))
  

class EncoderBlock(nn.Module):
  def __init__(self, self_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float) -> None:
    super().__init__()
    self.self_attention = self_attention
    self.feed_forward = feed_forward
    self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

  def forward(self, x, src_mask):
    x = self.residual_connections[0](x, lambda x_in: self.self_attention(x_in, x_in, x_in, src_mask))
    x = self.residual_connections[1](x, self.feed_forward)
    return x
  

class Encoder(nn.Module):
  def __init__(self, layers: nn.ModuleList) -> None:
    super().__init__()
    self.layers = layers
    self.norm = LayerNorm()

  def forward(self, x, mask):
    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)
  
class DecoderBlock(nn.Module):
  def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float) -> None:
    super().__init__()
    self.self_attention = self_attention
    self.cross_attention = cross_attention
    self.feed_forward = feed_forward
    self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

  def forward(self, x, encoder_output, src_mask, tgt_mask):
    x = self.residual_connections[0](x, lambda x_in: self.self_attention(x_in, x_in, x_in, tgt_mask))
    x = self.residual_connections[1](x, lambda x_in: self.cross_attention(x_in, encoder_output, encoder_output, src_mask))
    x = self.residual_connections[2](x, self.feed_forward)
    return x
  

class Decoder(nn.Module):
  def __init__(self, layers: nn.ModuleList) -> None:
    super().__init__()
    self.layers = layers
    self.norm = LayerNorm()

  def forward(self, x, encoder_output, src_mask, tgt_mask):
    for layer in self.layers:
      x = layer(x, encoder_output, src_mask, tgt_mask)
    return self.norm(x)
  

class ProjectionHead(nn.Module):
  def __init__(self, d_model: int, vocab_size:int) -> None:
    super().__init__()
    self.proj = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    return torch.log_softmax(self.proj(x), dim=-1)
  

class Transformer(nn.Module):
  def __init__(self, encoder: Encoder, decoder: Decoder, 
               src_embed: InputEmbedding, tgt_embed: InputEmbedding,
               pos_embed: PositionalEmbedding, proj: ProjectionHead) -> None:
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.pos_embed = pos_embed
    self.proj = proj

  def encode(self, src, src_mask):
    src = self.src_embed(src)
    src = self.pos_embed(src)
    return self.encoder(src, src_mask)
  
  def decode(self, encoder_output, src_mask, tgt, tgt_mask):
    tgt = self.tgt_embed(tgt)
    tgt = self.pos_embed(tgt)
    return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
  
  def project(self, x):
    return self.proj(x)
  

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int,
                      tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8,
                      dropout: float = 0.1, d_ff: int = 2048):
  # Create embedding layers
  src_embed = InputEmbedding(d_model, src_vocab_size)
  tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

  # Create positional embedding layer
  pos_embed = PositionalEmbedding(d_model, math.max(src_seq_len, tgt_seq_len))

  # Create encoder blocks
  encoder_blocks = []
  for _ in range(N):
    encoder_self_attention = MultiHeadAttention(d_model, h, dropout)
    feed_forward = FeedForward(d_model, d_ff, dropout)
    encoder_block = EncoderBlock(encoder_self_attention, feed_forward, dropout)
    encoder_blocks.append(encoder_block)

  # Create decoder blocks
  decoder_blocks = []
  for _ in range(N):
    decoder_self_attention = MultiHeadAttention(d_model, h, dropout)
    decoder_cross_attention = MultiHeadAttention(d_model, h, dropout)
    feed_forward = FeedForward(d_model, d_ff, dropout)
    decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, feed_forward, dropout)
    decoder_blocks.append(decoder_block)

  # Create the encoder, the decoder and the projection layer
  encoder = Encoder(nn.ModuleList(encoder_blocks))
  decoder = Decoder(nn.ModuleList(decoder_blocks))
  projection = ProjectionHead(d_model, tgt_vocab_size)

  # Create the transformer
  transformer = Transformer(encoder, decoder, src_embed, tgt_embed, pos_embed, projection)

  # Initialize model params with Xavier uniform
  for p in transformer.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)

  return transformer
