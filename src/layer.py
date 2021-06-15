import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Dropout, Embedding
from util import map_activation_str_to_layer

INF = 1e30
_INF = -1e30
EPS = 1e-8

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_mlp_layers=2, activation=None):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.num_mlp_layers = num_mlp_layers
        self.activation = activation
        self.output_dim = output_dim 

        if num_mlp_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for i in range(num_mlp_layers-2):
                self.bns.append(nn.BatchNorm1d(hidden_dim))
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        # init
        scale = 1/hidden_dim**0.5
        for layer in self.layers:
            nn.init.normal_(layer.weight, 0.0, scale)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        for i in range(self.num_mlp_layers-1):
            x = self.layers[i](x)
            x = self.bns[i](x)
            if self.activation:
                x = self.activation(x)
        return self.layers[-1](x)

    def get_output_dim(self):
        return self.output_dim


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FullyConnectedLayer, self).__init__()

        self.mlp = MLP(input_dim, hidden_dim, output_dim, num_mlp_layers=2, activation=None)

    def forward(self, x):
        return self.mlp(x)

    def get_output_dim(self):
        return self.mlp.output_dim


class Highway(nn.Module):
    def __init__(self, input_dim, num_layers=1, activation="relu"):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)])
        self.activation = map_activation_str_to_layer(activation)
        
        # init
        scale = 1/input_dim**0.5
        for layer in self.layers:
            nn.init.normal_(layer.weight, 0.0, scale)
            nn.init.constant_(layer.bias[:input_dim], 0.0)
            nn.init.constant_(layer.bias[input_dim:], 1.0)

    def forward(self, x):
        for layer in self.layers:
            o, g = layer(x).chunk(2, dim=-1)
            o = self.activation(o)
            g = F.sigmoid(g)
            x = g * x + (1 - g) * o
        return x


def multi_perspective_match(vector1, vector2, weight):
    assert vector1.size(0) == vector2.size(0)
    assert weight.size(1) == vector1.size(2)

    # (batch, seq_len, 1)
    similarity_single = F.cosine_similarity(vector1, vector2, 2).unsqueeze(2)

    # (1, 1, num_perspectives, hidden_size)
    weight = weight.unsqueeze(0).unsqueeze(0)

    # (batch, seq_len, num_perspectives, hidden_size)
    vector1 = weight * vector1.unsqueeze(2)
    vector2 = weight * vector2.unsqueeze(2)

    similarity_multi = F.cosine_similarity(vector1, vector2, dim=3)

    return similarity_single, similarity_multi


def multi_perspective_match_pairwise(vector1, vector2, weight):
    num_perspectives = weight.size(0)

    # (1, num_perspectives, 1, hidden_size)
    weight = weight.unsqueeze(0).unsqueeze(2)

    # (batch, num_perspectives, seq_len*, hidden_size)
    vector1 = weight * vector1.unsqueeze(1).expand(-1, num_perspectives, -1, -1)
    vector2 = weight * vector2.unsqueeze(1).expand(-1, num_perspectives, -1, -1)

    # (batch, num_perspectives, seq_len*, 1)
    vector1_norm = vector1.norm(p=2, dim=3, keepdim=True)
    vector2_norm = vector2.norm(p=2, dim=3, keepdim=True)

    # (batch, num_perspectives, seq_len1, seq_len2)
    mul_result = torch.matmul(vector1, vector2.transpose(2, 3))
    norm_value = vector1_norm * vector2_norm.transpose(2, 3)

    # (batch, seq_len1, seq_len2, num_perspectives)
    return (mul_result / norm_value.clamp(min=EPS)).permute(0, 2, 3, 1)


def masked_max(vector, mask, dim, keepdim=False):
    replaced_vector = vector.masked_fill(mask==0, _INF) if mask is not None else vector
    max_value, _ = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value


def masked_mean(vector, mask, dim, keepdim=False):
    replaced_vector = vector.masked_fill(mask==0, 0.0) if mask is not None else vector
    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask.float(), dim=dim, keepdim=keepdim)
    return value_sum / value_count.clamp(min=EPS)


def masked_softmax(vector, mask, dim=-1):
    if mask is None:
        result = F.softmax(vector, dim=dim)
    else:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        masked_vector = vector.masked_fill(mask==0, _INF)
        result = F.softmax(masked_vector, dim=dim)
    return result


class BiMpmMatching(nn.Module):
    def __init__(self,
                 hidden_dim,
                 num_perspectives,
                 share_weights_between_directions=True,
                 with_full_match=True,
                 with_maxpool_match=True,
                 with_attentive_match=True,
                 with_max_attentive_match=True):
        super(BiMpmMatching, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_perspectives = num_perspectives

        self.with_full_match = with_full_match
        self.with_maxpool_match = with_maxpool_match
        self.with_attentive_match = with_attentive_match
        self.with_max_attentive_match = with_max_attentive_match

        if not (with_full_match or with_maxpool_match or with_attentive_match or with_max_attentive_match):
            raise ValueError("At least one of the matching method should be enabled")

        def create_parameter():  # utility function to create and initialize a parameter
            param = nn.Parameter(torch.zeros(num_perspectives, hidden_dim))
            nn.init.kaiming_normal_(param)
            return param

        def share_or_create(weights_to_share):  # utility function to create or share the weights
            return weights_to_share if share_weights_between_directions else create_parameter()

        output_dim = 2  # used to calculate total output dimension, 2 is for cosine max and cosine min
        if with_full_match:
            self.full_forward_match_weights = create_parameter()
            self.full_forward_match_weights_reversed = share_or_create(self.full_forward_match_weights)
            self.full_backward_match_weights = create_parameter()
            self.full_backward_match_weights_reversed = share_or_create(self.full_backward_match_weights)
            output_dim += (num_perspectives + 1) * 2

        if with_maxpool_match:
            self.maxpool_match_weights = create_parameter()
            output_dim += num_perspectives * 2

        if with_attentive_match:
            self.attentive_match_weights = create_parameter()
            self.attentive_match_weights_reversed = share_or_create(self.attentive_match_weights)
            output_dim += num_perspectives + 1

        if with_max_attentive_match:
            self.max_attentive_match_weights = create_parameter()
            self.max_attentive_match_weights_reversed = share_or_create(self.max_attentive_match_weights)
            output_dim += num_perspectives + 1

        self.output_dim = output_dim

    def get_output_dim(self):
        return self.output_dim

    def forward(self, context_1, mask_1, context_2, mask_2):
        assert (not mask_2.requires_grad) and (not mask_1.requires_grad)
        assert context_1.size(-1) == context_2.size(-1) == self.hidden_dim

        # (batch,)
        len_1 = mask_1.sum(dim=1).long()
        len_2 = mask_2.sum(dim=1).long()

        # explicitly set masked weights to zero
        # (batch_size, seq_len*, hidden_dim)
        context_1 = context_1 * mask_1.unsqueeze(-1)
        context_2 = context_2 * mask_2.unsqueeze(-1)

        # array to keep the matching vectors for the two sentences
        matching_vector_1 = []
        matching_vector_2 = []

        # Step 0. unweighted cosine
        # First calculate the cosine similarities between each forward
        # (or backward) contextual embedding and every forward (or backward)
        # contextual embedding of the other sentence.

        # (batch, seq_len1, seq_len2)
        cosine_sim = F.cosine_similarity(context_1.unsqueeze(-2), context_2.unsqueeze(-3), dim=3)

        # (batch, seq_len*, 1)
        cosine_max_1 = masked_max(cosine_sim, mask_2.unsqueeze(-2), dim=2, keepdim=True)
        cosine_mean_1 = masked_mean(cosine_sim, mask_2.unsqueeze(-2), dim=2, keepdim=True)
        cosine_max_2 = masked_max(cosine_sim.permute(0, 2, 1), mask_1.unsqueeze(-2), dim=2, keepdim=True)
        cosine_mean_2 = masked_mean(cosine_sim.permute(0, 2, 1), mask_1.unsqueeze(-2), dim=2, keepdim=True)

        matching_vector_1.extend([cosine_max_1, cosine_mean_1])
        matching_vector_2.extend([cosine_max_2, cosine_mean_2])

        # Step 1. Full-Matching
        # Each time step of forward (or backward) contextual embedding of one sentence
        # is compared with the last time step of the forward (or backward)
        # contextual embedding of the other sentence
        if self.with_full_match:
            # (batch, 1, hidden_dim)
            last_position_1 = (len_1 - 1).clamp(min=0)
            last_position_1 = last_position_1.view(-1, 1, 1).expand(-1, 1, self.hidden_dim)
            last_position_2 = (len_2 - 1).clamp(min=0)
            last_position_2 = last_position_2.view(-1, 1, 1).expand(-1, 1, self.hidden_dim)

            context_1_forward_last = context_1.gather(1, last_position_1)
            context_2_forward_last = context_2.gather(1, last_position_2)
            context_1_backward_last = context_1[:, 0:1, :]
            context_2_backward_last = context_2[:, 0:1, :]

            # (batch, seq_len*, num_perspectives)
            matching_vector_1_forward_full = multi_perspective_match(context_1,
                                                                    context_2_forward_last,
                                                                    self.full_forward_match_weights)
            matching_vector_2_forward_full = multi_perspective_match(context_2,
                                                                    context_1_forward_last,
                                                                    self.full_forward_match_weights_reversed)
            matching_vector_1_backward_full = multi_perspective_match(context_1,
                                                                    context_2_backward_last,
                                                                    self.full_backward_match_weights)
            matching_vector_2_backward_full = multi_perspective_match(context_2,
                                                                    context_1_backward_last,
                                                                    self.full_backward_match_weights_reversed)

            matching_vector_1.extend(matching_vector_1_forward_full)
            matching_vector_1.extend(matching_vector_1_backward_full)
            matching_vector_2.extend(matching_vector_2_forward_full)
            matching_vector_2.extend(matching_vector_2_backward_full)

        # Step 2. Maxpooling-Matching
        # Each time step of forward (or backward) contextual embedding of one sentence
        # is compared with every time step of the forward (or backward)
        # contextual embedding of the other sentence, and only the max value of each
        # dimension is retained.
        if self.with_maxpool_match:
            # (batch, seq_len1, seq_len2, num_perspectives)
            matching_vector_max = multi_perspective_match_pairwise(context_1,
                                                                   context_2,
                                                                   self.maxpool_match_weights)

            # (batch, seq_len*, num_perspectives)
            matching_vector_1_max = masked_max(matching_vector_max,
                                               mask_2.unsqueeze(-2).unsqueeze(-1),
                                               dim=2)
            matching_vector_1_mean = masked_mean(matching_vector_max,
                                                 mask_2.unsqueeze(-2).unsqueeze(-1),
                                                 dim=2)
            matching_vector_2_max = masked_max(matching_vector_max.permute(0, 2, 1, 3),
                                               mask_1.unsqueeze(-2).unsqueeze(-1),
                                               dim=2)
            matching_vector_2_mean = masked_mean(matching_vector_max.permute(0, 2, 1, 3),
                                                 mask_1.unsqueeze(-2).unsqueeze(-1),
                                                 dim=2)

            matching_vector_1.extend([matching_vector_1_max, matching_vector_1_mean])
            matching_vector_2.extend([matching_vector_2_max, matching_vector_2_mean])


        # Step 3. Attentive-Matching
        # Each forward (or backward) similarity is taken as the weight
        # of the forward (or backward) contextual embedding, and calculate an
        # attentive vector for the sentence by weighted summing all its
        # contextual embeddings.
        # Finally match each forward (or backward) contextual embedding
        # with its corresponding attentive vector.

        # (batch, seq_len1, seq_len2, hidden_dim)
        att_2 = context_2.unsqueeze(-3) * cosine_sim.unsqueeze(-1)

        # (batch, seq_len1, seq_len2, hidden_dim)
        att_1 = context_1.unsqueeze(-2) * cosine_sim.unsqueeze(-1)

        if self.with_attentive_match:
            # (batch, seq_len*, hidden_dim)
            att_mean_2 = masked_softmax(att_2.sum(dim=2), mask_1.unsqueeze(-1))
            att_mean_1 = masked_softmax(att_1.sum(dim=1), mask_2.unsqueeze(-1))

            # (batch, seq_len*, num_perspectives)
            matching_vector_1_att_mean = multi_perspective_match(context_1,
                                                                 att_mean_2,
                                                                 self.attentive_match_weights)
            matching_vector_2_att_mean = multi_perspective_match(context_2,
                                                                 att_mean_1,
                                                                 self.attentive_match_weights_reversed)
            matching_vector_1.extend(matching_vector_1_att_mean)
            matching_vector_2.extend(matching_vector_2_att_mean)

        # Step 4. Max-Attentive-Matching
        # Pick the contextual embeddings with the highest cosine similarity as the attentive
        # vector, and match each forward (or backward) contextual embedding with its
        # corresponding attentive vector.
        if self.with_max_attentive_match:
            # (batch, seq_len*, hidden_dim)
            att_max_2 = masked_max(att_2, mask_2.unsqueeze(-2).unsqueeze(-1), dim=2)
            att_max_1 = masked_max(att_1.permute(0, 2, 1, 3), mask_1.unsqueeze(-2).unsqueeze(-1), dim=2)

            # (batch, seq_len*, num_perspectives)
            matching_vector_1_att_max = multi_perspective_match(context_1,
                                                                att_max_2,
                                                                self.max_attentive_match_weights)
            matching_vector_2_att_max = multi_perspective_match(context_2,
                                                                att_max_1,
                                                                self.max_attentive_match_weights_reversed)

            matching_vector_1.extend(matching_vector_1_att_max)
            matching_vector_2.extend(matching_vector_2_att_max)

        return matching_vector_1, matching_vector_2


class MultiHeadAttn(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, num_head,
            dropatt=0.0, 
            act_func="softmax", add_zero_attn=False, 
            pre_lnorm=False, post_lnorm=False):
        super(MultiHeadAttn, self).__init__()
        assert hidden_dim%num_head == 0

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.dropatt = nn.Dropout(dropatt)

        head_dim = hidden_dim // num_head

        self.q_net = nn.Linear(query_dim, hidden_dim, bias=False)
        self.k_net = nn.Linear(key_dim, hidden_dim, bias=False)
        self.v_net = nn.Linear(value_dim, hidden_dim, bias=False)
        self.o_net = nn.Linear(hidden_dim, query_dim, bias=False)

        self.scale = 1 / (head_dim ** 0.5)

        self.act_func = act_func
        self.add_zero_attn = add_zero_attn
        self.pre_lnorm = pre_lnorm
        self.post_lnorm = post_lnorm

        if pre_lnorm:
            self.q_layer_norm = nn.LayerNorm(query_dim)
            self.k_layer_norm = nn.LayerNorm(key_dim)
            self.v_layer_norm = nn.LayerNorm(value_dim)
        if post_lnorm:
            self.o_layer_norm = nn.LayerNorm(query_dim)
        
        # init
        for net in [self.q_net, self.k_net, self.v_net, self.o_net]:
            nn.init.xavier_uniform_(net.weight, 1.0)
            if hasattr(net, "bias") and net.bias is not None:
                nn.init.constant_(net.bias, 0.0)

        if self.pre_lnorm:
            for layer_norm in [self.q_layer_norm, self.k_layer_norm, self.v_layer_norm]:
                if hasattr(layer_norm, "weight"):
                    nn.init.normal_(layer_norm.weight, 1.0, self.scale)
                if hasattr(layer_norm, "bias") and layer_norm.bias is not None:
                    nn.init.constant_(layer_norm.bias, 0.0)
        if self.post_lnorm:
            if hasattr(self.o_layer_norm, "weight"):
                nn.init.normal_(self.o_layer_norm.weight, 1.0, self.scale)
            if hasattr(self.o_layer_norm, "bias") and self.o_layer_norm.bias is not None:
                nn.init.constant_(self.o_layer_norm.bias, 0.0)

    def forward(self, query, key, value, attn_mask=None):
        ##### multihead attention
        # [bsz x hlen x num_head x head_dim]
        bsz = query.size(0)

        if self.add_zero_attn:
            key = torch.cat([key, 
                torch.zeros((bsz, 1) + key.size()[2:], dtype=key.dtype, device=key.device)], dim=1)
            value = torch.cat([value, 
                torch.zeros((bsz, 1) + value.size()[2:], dtype=value.dtype, device=value.device)], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, 
                    torch.ones((bsz, 1), dtype=attn_mask.dtype, device=attn_mask.device)], dim=1)

        qlen, klen, vlen = query.size(1), key.size(1), value.size(1)

        if self.pre_lnorm:
            ##### layer normalization
            query = self.q_layer_norm(query)
            key = self.k_layer_norm(key)
            value = self.v_layer_norm(value)

        head_q = self.q_net(query).view(bsz, qlen, self.num_head, self.hidden_dim//self.num_head)
        head_k = self.k_net(key).view(bsz, klen, self.num_head, self.hidden_dim//self.num_head)
        head_v = self.v_net(value).view(bsz, vlen, self.num_head, self.hidden_dim//self.num_head)

        # [bsz x qlen x klen x num_head]
        attn_score = torch.einsum("bind,bjnd->bijn", (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_((attn_mask == 0).unsqueeze(1).unsqueeze(-1), _INF)
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_((attn_mask == 0).unsqueeze(-1), _INF)

        # [bsz x qlen x klen x num_head]
        if self.act_func is None or self.act_func == "None":
            attn_prob = attn_score
        elif self.act_func == "softmax":
            attn_prob = F.softmax(attn_score, dim=2)
        elif self.act_func == "sigmoid":
            attn_prob = F.sigmoid(attn_score)
        elif self.act_func == "tanh":
            attn_prob = F.tanh(attn_score)
        elif self.act_func == "relu":
            attn_prob = F.relu(attn_score)
        elif self.act_func == "leaky_relu":
            attn_prob = F.leaky_relu(attn_score)
        elif self.act_func == "maximum":
            max_score = torch.max(attn_score, dim=2, keepdim=True)[0]
            max_mask = attn_score == max_score
            cnt = torch.sum(max_mask, dim=2, keepdim=True)
            attn_prob = max_mask.float() / cnt.float()
        else:
            raise NotImplementedError
        attn_prob = self.dropatt(attn_prob)

        # [bsz x qlen x klen x num_head] x [bsz x klen x num_head x head_dim] -> [bsz x qlen x num_head x head_dim]
        attn_vec = torch.einsum("bijn,bjnd->bind", (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(bsz, qlen, self.hidden_dim)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        
        if self.post_lnorm:
            attn_out = self.o_layer_norm(attn_out)

        return attn_out

    def get_output_dim(self):
        return self.query_dim


class GatedMultiHeadAttn(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, num_head,
            dropatt=0.0, 
            act_func="softmax", add_zero_attn=False, 
            pre_lnorm=False, post_lnorm=False):
        super(GatedMultiHeadAttn, self).__init__()
        assert hidden_dim%num_head == 0

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.dropatt = nn.Dropout(dropatt)

        head_dim = hidden_dim // num_head

        self.q_net = nn.Linear(query_dim, hidden_dim, bias=False)
        self.k_net = nn.Linear(key_dim, hidden_dim, bias=False)
        self.v_net = nn.Linear(value_dim, hidden_dim, bias=False)
        self.o_net = nn.Linear(hidden_dim, query_dim, bias=False)
        self.g_net = nn.Linear(2*query_dim, query_dim, bias=True)

        self.scale = 1 / (head_dim ** 0.5)

        self.act_func = act_func
        self.add_zero_attn = add_zero_attn
        self.pre_lnorm = pre_lnorm
        self.post_lnorm = post_lnorm

        if pre_lnorm:
            self.q_layer_norm = nn.LayerNorm(query_dim)
            self.k_layer_norm = nn.LayerNorm(key_dim)
            self.v_layer_norm = nn.LayerNorm(value_dim)
        if post_lnorm:
            self.o_layer_norm = nn.LayerNorm(query_dim)
        
        # init
        for net in [self.q_net, self.k_net, self.v_net, self.o_net]:
            nn.init.xavier_uniform_(net.weight, 1.0)
            if hasattr(net, "bias") and net.bias is not None:
                nn.init.constant_(net.bias, 0.0)
        # when new data comes, it prefers to output 1 so that the gate is 1
        nn.init.normal_(self.g_net.weight, 0.0, self.scale)
        if hasattr(self.g_net, "bias") and self.g_net.bias is not None:
            nn.init.constant_(self.g_net.bias, 1.0)

        if self.pre_lnorm:
            for layer_norm in [self.q_layer_norm, self.k_layer_norm, self.v_layer_norm]:
                if hasattr(layer_norm, "weight"):
                    nn.init.normal_(layer_norm.weight, 1.0, self.scale)
                if hasattr(layer_norm, "bias") and layer_norm.bias is not None:
                    nn.init.constant_(layer_norm.bias, 0.0)
        if self.post_lnorm:
            if hasattr(self.o_layer_norm, "weight"):
                nn.init.normal_(self.o_layer_norm.weight, 1.0, self.scale)
            if hasattr(self.o_layer_norm, "bias") and self.o_layer_norm.bias is not None:
                nn.init.constant_(self.o_layer_norm.bias, 0.0)

    def forward(self, query, key, value, attn_mask=None):
        ##### multihead attention
        # [bsz x hlen x num_head x head_dim]
        bsz = query.size(0)

        if self.add_zero_attn:
            key = torch.cat([key, 
                torch.zeros((bsz, 1) + key.size()[2:], dtype=key.dtype, device=key.device)], dim=1)
            value = torch.cat([value, 
                torch.zeros((bsz, 1) + value.size()[2:], dtype=value.dtype, device=value.device)], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, 
                    torch.ones((bsz, 1), dtype=attn_mask.dtype, device=attn_mask.device)], dim=1)

        qlen, klen, vlen = query.size(1), key.size(1), value.size(1)

        if self.pre_lnorm:
            ##### layer normalization
            query = self.q_layer_norm(query)
            key = self.k_layer_norm(key)
            value = self.v_layer_norm(value)

        head_q = self.q_net(query).view(bsz, qlen, self.num_head, self.hidden_dim//self.num_head)
        head_k = self.k_net(key).view(bsz, klen, self.num_head, self.hidden_dim//self.num_head)
        head_v = self.v_net(value).view(bsz, vlen, self.num_head, self.hidden_dim//self.num_head)

        # [bsz x qlen x klen x num_head]
        attn_score = torch.einsum("bind,bjnd->bijn", (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_((attn_mask == 0).unsqueeze(1).unsqueeze(-1), _INF)
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_((attn_mask == 0).unsqueeze(-1), _INF)

        # [bsz x qlen x klen x num_head]
        if self.act_func is None or self.act_func == "None":
            attn_prob = attn_score
        elif self.act_func == "softmax":
            attn_prob = F.softmax(attn_score, dim=2)
        elif self.act_func == "sigmoid":
            attn_prob = F.sigmoid(attn_score)
        elif self.act_func == "tanh":
            attn_prob = F.tanh(attn_score)
        elif self.act_func == "relu":
            attn_prob = F.relu(attn_score)
        elif self.act_func == "leaky_relu":
            attn_prob = F.leaky_relu(attn_score)
        elif self.act_func == "maximum":
            max_score = torch.max(attn_score, dim=2, keepdim=True)[0]
            max_mask = attn_score == max_score
            cnt = torch.sum(max_mask, dim=2, keepdim=True)
            attn_prob = max_mask.float() / cnt.float()
        else:
            raise NotImplementedError
        attn_prob = self.dropatt(attn_prob)

        # [bsz x qlen x klen x num_head] x [bsz x klen x num_head x head_dim] -> [bsz x qlen x num_head x head_dim]
        attn_vec = torch.einsum("bijn,bjnd->bind", (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(bsz, qlen, self.hidden_dim)

        ##### linear projection
        attn_out = self.o_net(attn_vec)

        ##### gate
        gate = F.sigmoid(self.g_net(torch.cat([query, attn_out], dim=2)))
        attn_out = gate * query + (1-gate) * attn_out
        
        if self.post_lnorm:
            attn_out = self.o_layer_norm(attn_out)

        return attn_out

    def get_output_dim(self):
        return self.query_dim



class CnnHighway(nn.Module):
    def __init__(self, input_dim, filters, output_dim, num_highway=1, activation="relu", projection_location="after_highway", layer_norm=False):
        super().__init__()

        assert projection_location in ["after_cnn", "after_highway"]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection_location = projection_location

        self.activation = map_activation_str_to_layer(activation)
        # Create the convolutions
        self.convs = nn.ModuleList()
        for i, (width, num) in enumerate(filters):
            conv = nn.Conv1d(in_channels=input_dim, out_channels=num, kernel_size=width, bias=True)
            self.convs.append(conv)

        # Create the highway layers
        num_filters = sum(num for _, num in filters)
        if projection_location == 'after_cnn':
            highway_dim = output_dim
        else:
            # highway_dim is the number of cnn filters
            highway_dim = num_filters
        self.highways = Highway(highway_dim, num_highway, activation=activation)

        # Projection layer: always num_filters -> output_dim
        self.proj = nn.Linear(num_filters, output_dim)

        # And add a layer norm
        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

        # init
        scale = 1/num_filters**0.5
        for layer in self.convs:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        nn.init.normal_(self.proj.weight, 0.0, scale)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x, mask):

        # convolutions want (batch_size, input_dim, num_characters)
        x = x.transpose(1, 2)

        output = []
        for conv in self.convs:
            c = conv(x)
            c = torch.max(c, dim=-1)[0]
            c = self.activation(c)
            output.append(c)

        # (batch_size, n_filters)
        output = torch.cat(output, dim=-1)

        if self.projection_location == 'after_cnn':
            output = self.proj(output)

        # apply the highway layers (batch_size, highway_dim)
        output = self.highways(output)

        if self.projection_location == 'after_highway':
            # final projection  (batch_size, output_dim)
            output = self.proj(output)

        # apply layer norm if appropriate
        if self.layer_norm:
            output = self.layer_norm(output)

        return output

    def get_output_dim(self):
        return self.output_dim
