"""Assortment of R-net layers for use in models.py.

Author:
    Xiao Lu (shawlu@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import layers
from util import collate_fn, SQuAD
import util

class Encoder(nn.Module):
    def __init__(self, input_size, h_size, device, drop_prob=0):
        super(Encoder, self).__init__()
        """
        This encoder takes concatenated embeddings of context (or question)
        and output a single hidden state representing the context (or question)
        
        @param input_size (int): input size is the sum of word embedding size 
            and character-based word embedding size (output of CharEmbedding module)
            this number is kept as the standard reference in all subsequent layers
        @param h_size (int): size of hidden state, also the output size
        """
        self.input_size = input_size
        self.h_size = h_size
        self.device = device
        self.out_size = 2 * h_size
        self.gru = nn.GRU(input_size=input_size, 
            bidirectional=True, 
            hidden_size=h_size,
            batch_first=True)
        
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, input):
        """
        @param input (Tensor): concatenated embedding of character & word embeddings,
            shape (batch, sentence_length, embedding_size)
        @return last_hidden (Tensor): hidden state of every word of text, 
            shape (batch_size, sentence_length, 2 * h_size)
        """
        batch, sentence_length, embedding_size = input.size()
        
        # last_hidden: (batch_size, sentence_length, 2 * h_size)
        hiddens, last_hidden = self.gru(input) 

        # concatenate the output from two directions
        # last_hidden = torch.cat([last_hidden[0], last_hidden[1]], dim=1)
        
        # apply dropout
        return self.dropout(hiddens)
    
    
class GatedAttn(nn.Module):
    def __init__(self, input_size, h_size, device, drop_prob=0):
        super(GatedAttn, self).__init__()
        
        hidden_size = h_size
        self.hidden_size = hidden_size * 2
        self.input_size = input_size
        self.device = device
        self.out_size = 2 * h_size
        
        self.gru = nn.GRUCell(input_size=input_size * 2, hidden_size=h_size * 2)
        
        self.Wp = nn.Linear(in_features=input_size * 2, 
                            out_features=h_size * 2, 
                            bias=False)
        
        self.Wq = nn.Linear(in_features=input_size * 2, 
                            out_features=h_size * 2, 
                            bias=False)
        
        self.Wv = nn.Linear(in_features=h_size * 2, 
                            out_features=h_size * 2, 
                            bias=False)
        
        self.Wg = nn.Linear(in_features=input_size * 4, 
                            out_features=input_size * 4, 
                            bias=False)

        self.dropout = nn.Dropout(p=drop_prob)

    
    def forward(self, up, uq):
        up = up.permute(1, 0, 2) # [n, batch_size, 2 * h_size]
        uq = uq.permute(1, 0, 2) # [m, batch_size, 2 * h_size]
        (n, batch_size, _) = up.size()
        (m, _, _) = uq.size()

        Up = up
        Uq = uq
        
        vs = torch.zeros(n, batch_size, self.out_size).to(self.device)
        v = torch.randn(batch_size, self.hidden_size).to(self.device)
        V = torch.randn(batch_size, self.hidden_size, 1).to(self.device)
        
        # [64, 29, 400]
        Uq_ = Uq.permute([1, 0, 2])
        for i in range(n):
            Wup = self.Wp(Up[i]) # [64, 400] -> [64, h_size]
            Wuq = self.Wq(Uq) # [29, 64, 400] -> [29, 64, h_size]
            Wvv = self.Wv(v) # [batch_size, h_size] -> [batch_size, h_size]
            x = torch.tanh(Wup + Wuq + Wvv) # (29, 64, 400)
            
            x = x.permute([1, 0, 2]) # (64, 29, 400)
            s = torch.bmm(x, V).squeeze() # (64, 29)
            
            a = torch.softmax(s, dim=1).unsqueeze(1) # [64, 1, 29]
            c = torch.bmm(a, Uq_).squeeze() # (64, 400)
  
            r = torch.cat([Up[i], c], dim=1) # (64, 800)
            g = torch.sigmoid(self.Wg(r)) # (64, 800)
            r_ = torch.mul(g, r) # element-wise mult
            
            c_ = r_[:, self.input_size*2:]
            v = self.gru(c_, v)
            vs[i] = v
            del Wup, Wuq, Wvv, x, a, s, c, g, r, r_, c_
        del up, uq, Up, Uq, Uq_
        vs = self.dropout(vs)
        return vs

class SelfAttn(nn.Module):
    def __init__(self, in_size, device, drop_prob=0):
        super(SelfAttn, self).__init__()
        self.hidden_size = in_size
        self.in_size = in_size
        self.device = device
        
        self.gru = nn.GRUCell(input_size=in_size, hidden_size=self.hidden_size)
        
        self.Wp = nn.Linear(self.in_size, self.hidden_size, bias=False)
        self.Wp_ = nn.Linear(self.in_size, self.hidden_size, bias=False)
        
        self.out_size = self.hidden_size
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, v):
        (l, batch_size, _) = v.size()
        h = torch.randn(batch_size, self.hidden_size).to(self.device)
        V = torch.randn(batch_size, self.hidden_size, 1).to(self.device)
        hs = torch.zeros(l, batch_size, self.out_size).to(self.device)
        
        for i in range(l):
            Wpv = self.Wp(v[i])
            Wpv_ = self.Wp_(v)
            x = torch.tanh(Wpv + Wpv_)
            x = x.permute([1, 0, 2])
            s = torch.bmm(x, V)
            s = torch.squeeze(s, 2)
            a = torch.softmax(s, 1).unsqueeze(1)
            c = torch.bmm(a, v.permute([1, 0, 2])).squeeze()
            h = self.gru(c, h)
            hs[i] = h
            del Wpv, Wpv_, x, s, a, c
        hs = self.dropout(hs)
        del h, v
        return hs

class Pointer(nn.Module):
    def __init__(self, in_size1, in_size2, device):
        super(Pointer, self).__init__()
        self.hidden_size = in_size2
        self.in_size1 = in_size1
        self.in_size2 = in_size2
        self.device = device
        self.gru = nn.GRUCell(input_size=in_size1, hidden_size=self.hidden_size)
        # Wu uses bias. See formula (11). Maybe Vr is just a bias.
        self.Wu = nn.Linear(self.in_size2, self.hidden_size, bias=True)
        self.Wh = nn.Linear(self.in_size1, self.hidden_size, bias=False)
        self.Wha = nn.Linear(self.in_size2, self.hidden_size, bias=False)
        self.out_size = 1

    def forward(self, h, u):
        """
        self matching output, 
        Uq: [64, 29, 400]
        """
        (lp, batch_size, _) = h.size()
  
        u = u.permute(1, 0, 2)
        (lq, _, _) = u.size()
        v = torch.randn(batch_size, self.hidden_size, 1).to(self.device)
        u_ = u.permute([1,0,2]) # (m, batch_size, h_size * 2)
        h_ = h.permute([1,0,2]) # (m, batch_size, h_size * 2)
        x = torch.tanh(self.Wu(u)).permute([1, 0, 2])
        s = torch.bmm(x, v)
        s = torch.squeeze(s, 2)
        a = torch.softmax(s, 1).unsqueeze(1)
        r = torch.bmm(a, u_).squeeze()
        x = torch.tanh(self.Wh(h)+self.Wha(r)).permute([1, 0, 2])
        s = torch.bmm(x, v)
        s = torch.squeeze(s)
        p1 = torch.softmax(s, 1)
        c = torch.bmm(p1.unsqueeze(1), h_).squeeze()
        r = self.gru(c, r)
        x = torch.tanh(self.Wh(h) + self.Wha(r)).permute([1, 0, 2])
        s = torch.bmm(x, v)
        s = torch.squeeze(s)
        p2 = torch.softmax(s, 1)
        return p1, p2