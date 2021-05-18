import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from collections import OrderedDict
from transformers import BertModel, RobertaModel
from layer import *

class Encoder(nn.Module):
    def __init__(self, **kw):
        """
        Encoder that encodes raw sentence ids into token-level vectors (for BERT)
        """
        super(Encoder, self).__init__()

    def set_finetune(self, finetune):
        raise NotImplementedError

    def forward(self, x, mask=None):
        raise NotImplementedError

    def get_output_dim(self):
        raise NotImplementedError


class LSTMEncoder(Encoder):
    def __init__(self, **kw):
        super(LSTMEncoder, self).__init__(**kw)
        word2vec_file = kw.get("word2vec_file", "../data/model/glove/glove_wsj.txt")
        word_dim = kw.get("word_dim", 300)
        hidden_dim = kw.get("hidden_dim", 128)
        num_lstm_layers = kw.get("num_lstm_layers", 1)
        dropout = kw.get("dropout", 0.2)

        # embedding layers
        word2idx = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
        with open(word2vec_file, "r") as f:
            num_words, word_dim = f.readline().split()
            num_words, word_dim = int(num_words), int(word_dim)
            # two special tokens <pad>, <unk> not in the file
            weight = np.zeros((num_words+2, word_dim), dtype=np.float32)
            for line in f:
                word, vec = line.split(" ", 1)
                idx = word2idx.get(word, -1)
                if idx == -1:
                    idx = len(word2idx)
                    word2idx[word] = idx
                weight[idx] = np.array(vec.strip().split(), dtype=np.float32)
        weight[1] = np.mean(weight[2:], axis=0)

        self.word_embedding = Embedding(
            num_embeddings=num_words+2, 
            embedding_dim=word_dim,
            padding_idx=0,
            _weight=torch.from_numpy(weight))
        self.token_type_embeddings = Embedding(2, word_dim)
        
        self.bilstm = LSTM(input_size=word_dim, hidden_size=hidden_dim, 
            num_lstm_layers=num_lstm_layers, bidirectional=True,
            dropout=dropout, batch_first=True)
        self.set_finetune("full")
    
    def set_finetune(self, finetune):
        if finetune == "none":
            for param in self.parameters():
                param.requires_grad = False
        elif finetune == "full":
            for param in self.parameters():
                param.requires_grad = True
        elif finetune == "last":
            for param in self.parameters():
                param.requires_grad = False
            last_layer_key = "_l%d" % (self.bilstm.num_lstm_layers-1)
            for name, param in self.bilstm.named_parameters():
                if name.endswith(last_layer_key):
                    param.requires_grad = True
        elif finetune == "type":
            for param in self.parameters():
                param.requires_grad = False
            self.token_type_embeddings.weight.requires_grad = True
        self.word_embedding.weight.requires_grad = False

    def forward(self, x, mask=None):
        return self.forward_single(x, mask)

    def forward_single(self, x, mask=None):
        bsz = x.size(0)
        x_emb = self.word_embedding(x)
        # we assume type_emb is 0

        output, (h, c) = self.bilstm(x_emb)
        return output, mask

    def forward_pair(self, x1, x2, mask1=None, mask2=None):
        bsz = x1.size(0)
        x1_len, x2_len = x1.size(1), x2.size(1)
        
        x = torch.cat([x1, x2], dim=1)

        x_emb = self.word_embedding(x)
        type_ids = torch.empty_like(x)
        type_ids[:, :x1_len].data.fill_(0)
        type_ids[:, x1_len:].data.fill_(1)
        type_emb = self.token_type_embeddings(type_ids)
        x_emb = x_emb + type_emb

        x_output, (h, c) = self.bilstm(x_emb)

        return x_output[:, :x1_len], x_output[:, x1_len:], mask1, mask2
    
    def get_output_dim(self):
        return self.bilstm.hidden_size * 2


class BERTEncoder(Encoder):
    def __init__(self, **kw):
        super(BERTEncoder, self).__init__(**kw)
        bert_dir = kw.get("bert_dir", "../data/pretrained_lm/bert")

        self.model = BertModel.from_pretrained("bert-base-uncased", cache_dir=bert_dir)
        embedding_dim = self.model.embeddings.word_embeddings.embedding_dim
        num_position_embeddings = self.model.embeddings.position_embeddings.num_embeddings
        num_segments = kw.get("num_segments", 2)
        max_len = kw.get("max_len", 512)

        with torch.no_grad():
            # position embeddings
            r = math.ceil(max_len/num_position_embeddings)
            new_position_embeddings = nn.Embedding(r*num_position_embeddings, embedding_dim)
            new_position_embeddings.weight.data.copy_(self.model.embeddings.position_embeddings.weight.repeat(r, 1))
            del self.model.embeddings.position_embeddings
            self.model.embeddings.position_embeddings = new_position_embeddings

            # segment embeddings
            r = math.ceil(num_segments/self.model.embeddings.token_type_embeddings.num_embeddings)
            new_token_type_embeddings = nn.Embedding(r*self.model.embeddings.token_type_embeddings.num_embeddings, 768)
            new_token_type_embeddings.weight.data.copy_(self.model.embeddings.token_type_embeddings.weight.repeat(r, 1))
            del self.model.embeddings.token_type_embeddings
            self.model.embeddings.token_type_embeddings = new_token_type_embeddings

        self.set_finetune("full")
        
    def set_finetune(self, finetune):
        if finetune == "none":
            for param in self.model.parameters():
                param.requires_grad = False
        elif finetune == "full":
            for param in self.model.parameters():
                param.requires_grad = True
        elif finetune == "last":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.encoder.layer[-1].parameters():
                param.requires_grad = True
        elif finetune == "type":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.embeddings.token_type_embeddings.parameters():
                param.requires_grad = True

    def forward(self, x, mask=None):
        return self.forward_single(x, mask)
    
    def forward_single(self, x, mask=None):
        last_hidden_state = self.model(x, attention_mask=mask)[0]
        return last_hidden_state, mask

    def forward_pair(self, x1, x2, mask1=None, mask2=None):
        bsz = x1.size(0)
        x1_len, x2_len = x1.size(1), x2.size(1)
        
        x = torch.cat([x1, x2[:, 1:]], dim=1)
        type_ids = torch.empty_like(x)
        type_ids[:, :x1_len].data.fill_(0)
        type_ids[:, x1_len:].data.fill_(1)
        if mask1 is not None and mask2 is not None:
            mask = torch.cat([mask1, mask2[:, 1:]], dim=1)
        else:
            mask = None

        x_output = self.model(x, attention_mask=mask, token_type_ids=type_ids)[0]

        return x_output[:, :x1_len], x_output[:, x1_len:], mask1, mask2[:, 1:] if mask2 is not None else None

    def get_output_dim(self):
        return 768


class ROBERTAEncoder(Encoder):
    def __init__(self, **kw):
        super(ROBERTAEncoder, self).__init__(**kw)
        roberta_dir = kw.get("roberta_dir", "../data/pretrained_lm/roberta")

        self.model = RobertaModel.from_pretrained("roberta-base", cache_dir=roberta_dir)
        embedding_dim = self.model.embeddings.word_embeddings.embedding_dim
        num_position_embeddings = self.model.embeddings.position_embeddings.num_embeddings
        num_segments = kw.get("num_segments", 2)
        max_len = kw.get("max_len", 512)

        with torch.no_grad():
            # position embeddings
            r = math.ceil(max_len/num_position_embeddings)
            new_position_embeddings = nn.Embedding(r*num_position_embeddings, embedding_dim)
            new_position_embeddings.weight.data.copy_(self.model.embeddings.position_embeddings.weight.repeat(r, 1))
            del self.model.embeddings.position_embeddings
            self.model.embeddings.position_embeddings = new_position_embeddings

            # segment embeddings
            r = math.ceil(num_segments/self.model.embeddings.token_type_embeddings.num_embeddings)
            new_token_type_embeddings = nn.Embedding(r*self.model.embeddings.token_type_embeddings.num_embeddings, 768)
            new_token_type_embeddings.weight.data.copy_(self.model.embeddings.token_type_embeddings.weight.repeat(r, 1))
            del self.model.embeddings.token_type_embeddings
            self.model.embeddings.token_type_embeddings = new_token_type_embeddings
        self.set_finetune("full")

    def set_finetune(self, finetune):
        if finetune == "none":
            for param in self.model.parameters():
                param.requires_grad = False
        elif finetune == "full":
            for param in self.model.parameters():
                param.requires_grad = True
        elif finetune == "last":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.encoder.layer[-1].parameters():
                param.requires_grad = True
        elif finetune == "type":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.embeddings.token_type_embeddings.parameters():
                param.requires_grad = True

    def forward(self, x, mask=None):
        return self.forward_single(x, mask)
    
    def forward_single(self, x, mask=None):
        last_hidden_state = self.model(x, attention_mask=mask)[0]
        return last_hidden_state, mask

    def forward_pair(self, x1, x2, mask1=None, mask2=None):
        bsz = x1.size(0)
        x1_len, x2_len = x1.size(1), x2.size(1)
        
        sep = torch.empty([bsz, 1], dtype=torch.long, device=x1.device)
        sep.data.fill_(2) # 2 is the id for </s>

        x = torch.cat([x1, sep, x2[:, 1:]], dim=1)
        type_ids = torch.empty_like(x)
        type_ids[:, :x1_len].data.fill_(0)
        type_ids[:, x1_len:].data.fill_(1)
        if mask1 is not None and mask2 is not None:
            mask = torch.cat([mask1, mask2], dim=1)
        else:
            mask = None

        x_output = self.model(x, attention_mask=mask, token_type_ids=type_ids)[0]
        return x_output[:, :x1_len], x_output[:, x1_len:], mask1, mask2

    def get_output_dim(self):
        return 768
