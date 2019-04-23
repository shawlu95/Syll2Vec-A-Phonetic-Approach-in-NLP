"""Top-level model classes.

Author:
    Xiao Lu (shawlu@stanford.edu)
    Chris Chute (chute@stanford.edu)
"""

import layers
from layers_rnet import Encoder, GatedAttn, SelfAttn, Pointer
import torch
import torch.nn as nn

class RNet(nn.Module):
    def __init__(self, word_vectors, char_vectors, device, hidden_size, drop_prob=0.):
        super(RNet, self).__init__()
        self.device = device
        self.word_emb = nn.Embedding.from_pretrained(word_vectors)
        self.char_emb = layers.CharEmbedding(char_vectors=char_vectors, 
                                             e_char=char_vectors.size(1), 
                                             e_word=word_vectors.size(1), 
                                             drop_prob=drop_prob, 
                                             freeze=False)
        
        self.proj = nn.Linear(word_vectors.size(1) * 2, hidden_size, bias=False)

        self.hwy = layers.HighwayEncoder(2, hidden_size)

        self.encoder = Encoder(input_size=hidden_size, 
                               h_size=hidden_size, 
                               device=device, 
                               drop_prob=drop_prob)

        self.gatedAttn = GatedAttn(input_size=hidden_size, 
                                   h_size=hidden_size, 
                                   device=device, 
                                   drop_prob=drop_prob)

        self.selfAttn = SelfAttn(self.gatedAttn.out_size, 
                                 device=device, 
                                 drop_prob=drop_prob)

        self.pointer = Pointer(self.selfAttn.out_size, 
                               self.encoder.out_size, 
                               device=device)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        
        qc = self.char_emb.forward(qc_idxs)
        cc = self.char_emb.forward(cc_idxs)

        qw = self.word_emb(qw_idxs)
        cw = self.word_emb(cw_idxs)

        q_emb = torch.cat((qc, qw), dim=2)
        c_emb = torch.cat((cc, cw), dim=2)

        q_emb = self.proj.forward(q_emb)  # (batch_size, seq_len, hidden_size)
        q_emb = self.hwy.forward(q_emb)   # (batch_size, seq_len, hidden_size)

        c_emb = self.proj.forward(c_emb)  # (batch_size, seq_len, hidden_size)
        c_emb = self.hwy.forward(c_emb)   # (batch_size, seq_len, hidden_size)

        uc = self.encoder.forward(c_emb)
        uq = self.encoder.forward(q_emb)

        v = self.gatedAttn.forward(uc, uq)

        h = self.selfAttn.forward(v)
        p1, p2 = self.pointer.forward(h, uq)
        return p1, p2

class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, use_char=False, char_vectors=None, 
        use_syll=False, syll_vectors=None, drop_prob=0.):

        super(BiDAF, self).__init__()
        self.word_emb_size = word_vectors.size(1)

        self.emb = layers.WordEmbedding(word_vectors=word_vectors,
                                        hidden_size=hidden_size,
                                        drop_prob=drop_prob)

        self.use_char = use_char
        self.use_syll = use_syll

        if use_char and use_syll:
            self.char_emb = layers.CharEmbedding(char_vectors, 
                                                 e_char=char_vectors.size(1), 
                                                 e_word=hidden_size, 
                                                 drop_prob=drop_prob,
                                                 freeze=False)
            self.syll_emb = layers.SyllEmbedding(syll_vectors, 
                                                 e_syll=syll_vectors.size(1), 
                                                 e_word=hidden_size, 
                                                 drop_prob=drop_prob,
                                                 freeze=False)
            self.input_size = self.word_emb_size + 2 * hidden_size
        elif use_char:
            self.char_emb = layers.CharEmbedding(char_vectors, 
                                                 e_char=char_vectors.size(1), 
                                                 e_word=hidden_size, 
                                                 drop_prob=drop_prob,
                                                 freeze=False)
            self.input_size = self.word_emb_size + hidden_size
        elif use_syll:
            self.syll_emb = layers.SyllEmbedding(syll_vectors, 
                                                 e_syll=syll_vectors.size(1), 
                                                 e_word=hidden_size, 
                                                 drop_prob=drop_prob,
                                                 freeze=False)
            self.input_size = self.word_emb_size + hidden_size
        else:
            self.input_size = self.word_emb_size


        self.proj = nn.Linear(self.input_size, hidden_size, bias=False)
        self.hwy = layers.HighwayEncoder(2, hidden_size)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                             hidden_size=hidden_size,
                             num_layers=1,
                             drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs=None, qc_idxs=None, cs_idxs=None, qs_idxs=None):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        cw_emb = self.emb.forward(cw_idxs)         # (batch_size, c_len, word_emb_size)
        qw_emb = self.emb.forward(qw_idxs)         # (batch_size, q_len, word_emb_size)

        if self.use_char and self.use_syll:
            cc_emb = self.char_emb.forward(cc_idxs) # (batch_size, c_len, hidden_size)
            qc_emb = self.char_emb.forward(qc_idxs) #  (batch_size, q_len, hidden_size)

            cs_emb = self.syll_emb.forward(cs_idxs) # (batch_size, c_len, hidden_size)
            qs_emb = self.syll_emb.forward(qs_idxs) # (batch_size, q_len, hidden_size)

            c_emb = torch.cat((cw_emb, cc_emb, cs_emb), dim=2) # (batch_size, c_len, 2 * hidden_size + word_emb_size)
            q_emb = torch.cat((qw_emb, qc_emb, qs_emb), dim=2) # (batch_size, q_len, 2 * hidden_size + word_emb_size)

        elif self.use_char:
            cc_emb = self.char_emb.forward(cc_idxs) # (batch_size, c_len, hidden_size + word_emb_size)
            qc_emb = self.char_emb.forward(qc_idxs) # (batch_size, q_len, hidden_size + word_emb_size)

            c_emb = torch.cat((cw_emb, cc_emb), dim=2) # (batch_size, c_len, hidden_size + word_emb_size)
            q_emb = torch.cat((qw_emb, qc_emb), dim=2) # (batch_size, q_len, hidden_size + word_emb_size)
            
        elif self.use_syll:
            cs_emb = self.syll_emb.forward(cs_idxs) # (batch_size, c_len, hidden_size)
            qs_emb = self.syll_emb.forward(qs_idxs) # (batch_size, q_len, hidden_size)

            c_emb = torch.cat((cw_emb, cs_emb), dim=2) # (batch_size, c_len, word_emb_size + word_emb_size)
            q_emb = torch.cat((qw_emb, qs_emb), dim=2) # (batch_size, q_len, word_emb_size + word_emb_size)

        c_emb = self.proj.forward(c_emb)  # (batch_size, seq_len, hidden_size)
        c_emb = self.hwy.forward(c_emb)   # (batch_size, seq_len, hidden_size)

        q_emb = self.proj.forward(q_emb)  # (batch_size, seq_len, hidden_size)
        q_emb = self.hwy.forward(q_emb)   # (batch_size, seq_len, hidden_size)

        c_enc = self.enc.forward(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc.forward(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att.forward(c_enc, q_enc,
                               c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod.forward(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out.forward(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
