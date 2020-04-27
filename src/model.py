import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from collections import OrderedDict
from encoder import *
from layers import *

class BMGFModel(nn.Module):
    def __init__(self, **kw):
        super(BMGFModel, self).__init__()
        min_arg = kw.get("min_arg", 3)
        encoder = kw.get("encoder", "roberta")
        hidden_dim = kw.get("hidden_dim", 128)
        num_perspectives = kw.get("num_perspectives", 16)
        dropout = kw.get("dropout", 0.2)
        activation = kw.get("activation", "relu")
        num_rels = kw.get("num_rels", 4)
        filters = kw.get("filters", [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]])
        num_filters = kw.get("num_filters", 64)
        act_layer = map_activation_str_to_layer(activation)

        self.drop = Dropout(dropout)
        if encoder == "lstm":
            self.encoder = LSTMEncoder(**kw)
        elif encoder == "bert":
            self.encoder = BERTEncoder(**kw)
        elif encoder == "roberta":
            self.encoder = ROBERTAEncoder(**kw)
        else:
            raise NotImplementedError("Error: encoder=%s is not supported now." % (encoder))

        self.bimpm = BiMpmMatching(
            hidden_dim=self.encoder.get_output_dim(), 
            num_perspectives=num_perspectives)
        output_dim = self.encoder.get_output_dim() + self.bimpm.get_output_dim()

        self.gated_attn_layer = GatedMultiHeadAttn(
            query_dim=output_dim,
            key_dim=output_dim,
            value_dim=output_dim,
            hidden_dim=hidden_dim,
            num_head=num_perspectives,
            dropatt=dropout,
            act_func="softmax",
            add_zero_attn=False,
            pre_lnorm=False,
            post_lnorm=False)

        self.conv_layer = CnnHighway(
            input_dim=self.gated_attn_layer.get_output_dim(),
            output_dim=hidden_dim,
            filters=[(1, num_filters), (2, num_filters)], # the shortest length is 2
            num_highway=1,
            activation=activation,
            layer_norm=False)

        self.fc_layer = FullyConnectedLayer(2 * hidden_dim, hidden_dim, num_rels)

    def set_finetune(self, finetune):
        for param in self.parameters():
            param.requires_grad = True
        self.encoder.set_finetune(finetune)
    
    def forward(self, arg1, arg2, arg1_mask=None, arg2_mask=None, encode_pair=True):
        if encode_pair:
            arg1_feats, arg2_feats, arg1_mask, arg2_mask = self.encoder.forward_pair(arg1, arg2, arg1_mask, arg2_mask)
        else:
            arg1_feats, arg1_mask = self.encoder.forward_single(arg1, arg1_mask)
            arg2_feats, arg2_mask = self.encoder.forward_single(arg2, arg2_mask)
        arg1_feats, arg2_feats = self.drop(arg1_feats), self.drop(arg2_feats)

        arg1_matched_feats, arg2_matched_feats = self.bimpm(
            arg1_feats, arg1_mask, arg2_feats, arg2_mask)
        arg1_matched_feats = torch.cat(arg1_matched_feats, dim=2)
        arg2_matched_feats = torch.cat(arg2_matched_feats, dim=2)
        arg1_matched_feats, arg2_matched_feats = self.drop(arg1_matched_feats), self.drop(arg2_matched_feats)

        arg1_self_attned_feats = torch.cat([arg1_feats, arg1_matched_feats], dim=2)
        arg2_self_attned_feats = torch.cat([arg2_feats, arg2_matched_feats], dim=2)
        arg1_self_attned_feats = self.gated_attn_layer(
            arg1_self_attned_feats, arg1_self_attned_feats, arg1_self_attned_feats, attn_mask=arg1_mask)
        arg2_self_attned_feats = self.gated_attn_layer(
            arg2_self_attned_feats, arg2_self_attned_feats, arg2_self_attned_feats, attn_mask=arg2_mask)
        arg1_self_attned_feats, arg2_self_attned_feats = self.drop(arg1_self_attned_feats), self.drop(arg2_self_attned_feats)

        arg1_conv = self.conv_layer(arg1_self_attned_feats, arg1_mask)
        arg2_conv = self.conv_layer(arg2_self_attned_feats, arg2_mask)
        arg1_conv, arg2_conv = self.drop(arg1_conv), self.drop(arg2_conv)
        
        output = self.fc_layer(torch.cat([arg1_conv, arg2_conv], dim=1))

        return output # unnormalized results
