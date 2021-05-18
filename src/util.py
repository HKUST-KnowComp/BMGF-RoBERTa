import os
import socket
import re
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from stanfordnlp.server import CoreNLPClient

##############################
##### argument functions #####
##############################
def str2value(x):
    return eval(x)

def str2bool(x):
    x = x.lower()
    return x == "true" or x == "yes"

def str2list(x):
    results = []
    for x in x.split(","):
        x = x.strip()
        try:
            x = str2value(x)
        except:
            pass
        results.append(x)
    return results

##############################
###### stanford corenlp ######
##############################
def is_port_occupied(ip='127.0.0.1', port=80):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        return True
    except:
        return False

def get_corenlp_client(corenlp_path, corenlp_port):
    annotators = ["tokenize", "ssplit"]

    os.environ["CORENLP_HOME"] = corenlp_path
    if is_port_occupied(port=corenlp_port):
        try:
            corenlp_client = CoreNLPClient(
                annotators=annotators, timeout=99999,
                memory='4G', endpoint="http://localhost:%d" % corenlp_port,
                start_server=False, be_quiet=False)
            return corenlp_client
        except Exception as err:
            raise err
    else:
        print("Starting corenlp client at port {}".format(corenlp_port))
        corenlp_client = CoreNLPClient(
            annotators=annotators, timeout=99999,
            memory='4G', endpoint="http://localhost:%d" % corenlp_port,
            start_server=True, be_quiet=False)
        return corenlp_client

def sentence_split_with_corenlp(sentence, corenlp_client):
    results = list()
    while len(results) == 0:
        try:
            for sent in corenlp_client.annotate(sentence, annotators=["ssplit"], output_format="json")["sentences"]:
                if sent['tokens']:
                    char_st = sent['tokens'][0]['characterOffsetBegin']
                    char_end = sent['tokens'][-1]['characterOffsetEnd']
                else:
                    char_st, char_end = 0, 0
                results.append(sentence[char_st:char_end])
            break
        except:
            pass
    return results

def tokenize_with_corenlp(sentence, corenlp_client):
    results = list()
    
    while len(results) == 0:
        try:
            for sent in corenlp_client.annotate(sentence, annotators=["ssplit"], output_format="json")["sentences"]:
                results.append([t['word'] for t in sent['tokens']])
            break
        except:
            pass
    return results


##############################
######## os functions ########
##############################
def save_config(config, path):
    with open(path, "w") as f:
        json.dump(vars(config), f)

def load_config(path):
    with open(path, "r") as f:
        config = json.load(f, object_hook=lambda d: namedtuple('config', d.keys())(*d.values()))
    return config

def _map_tensor_to_list(tensor):
    return tensor.tolist()

def _map_array_to_list(array):
    return array.tolist()

def _map_list_to_python_type(l):
    if len(l) == 0:
        return l
    if isinstance(l[0], dict):
        return [_map_dict_to_python_type(x) for x in l]
    elif isinstance(l[0], list):
        return [_map_list_to_python_type(x) for x in l]
    elif isinstance(l[0], torch.Tensor):
        return [_map_tensor_to_list(x) for x in l]
    elif isinstance(l[0], np.ndarray):
        return [_map_array_to_list(x) for x in l]
    else:
        return l

def _map_dict_to_python_type(d):
    new_d = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            new_d[k] = _map_dict_to_python_type(v)
        elif isinstance(v, list):
            new_d[k] = _map_list_to_python_type(v)
        elif isinstance(v, torch.Tensor):
            new_d[k] = _map_tensor_to_list(v)
        elif isinstance(v, np.ndarray):
            new_d[k] = _map_array_to_list(v)
        else:
            new_d[k] = v
    return new_d

def save_results(results, path):
    with open(path, "w") as f:
        json.dump(_map_dict_to_python_type(results), f)

def get_best_epochs(log_file, by="loss"):
    regex = re.compile(r"data_type:\s+(\w+)\s+best\s+([\w\-]+).*?\(epoch:\s+(\d+)\)")
    best_epochs = dict()
    # get the best epoch
    try:
        lines = subprocess.check_output(["tail", log_file, "-n12"]).decode("utf-8").split("\n")[0:-1]
    except:
        with open(log_file, "r") as f:
            lines = f.readlines()
    
    for line in lines[-12:]:
        matched_results = regex.findall(line)
        for matched_result in matched_results:
            if by in matched_result[1]:
                best_epochs[matched_result[0]] = int(matched_result[2])
    if len(best_epochs) == 0:
        for line in lines:
            matched_results = regex.findall(line)
            if by in matched_result[1]:
                best_epochs[matched_result[0]] = int(matched_result[2])
    return best_epochs

def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


##############################
### deep learning functions ##
##############################
def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes):
    bsz = len(graph_sizes)
    dim, dtype, device = batched_graph_feats.size(-1), batched_graph_feats.dtype, batched_graph_feats.device

    min_size, max_size = min(graph_sizes), max(graph_sizes)
    mask = torch.ones((bsz, max_size), dtype=torch.long, device=device, requires_grad=False)

    if min_size == max_size:
        return batched_graph_feats.view(bsz, max_size, -1), mask
    else:
        unbatched_graph_feats = list(torch.split(batched_graph_feats, graph_sizes, dim=0))
        for i, l in enumerate(graph_sizes):
            if l == max_size:
                continue
            elif l > max_size:
                unbatched_graph_feats[i] = unbatched_graph_feats[i][:max_size]
            else:
                mask[i, l:].fill_(0)
                zeros = torch.zeros((max_size-l, dim), dtype=dtype, device=device, requires_grad=False)
                unbatched_graph_feats[i] = torch.cat([unbatched_graph_feats[i], zeros], dim=0)
        return torch.stack(unbatched_graph_feats, dim=0), mask

def batch_convert_list_to_tensor(batch_list, max_seq_len=-1):
    batch_tensor = [torch.tensor(v) for v in batch_list]
    return batch_convert_tensor_to_tensor(batch_tensor)

def batch_convert_tensor_to_tensor(batch_tensor, max_seq_len=-1):
    batch_lens = [len(v) for v in batch_tensor]
    if max_seq_len == -1:
        max_seq_len = max(batch_lens)

    result = torch.ones([len(batch_tensor), max_seq_len] + list(batch_tensor[0].size())[1:], dtype=batch_tensor[0].dtype, requires_grad=False)
    for i, t in enumerate(batch_tensor):
        len_t = batch_lens[i]
        if len_t < max_seq_len:
            result[i, :len_t].data.copy_(t)
        elif len_t == max_seq_len:
            result[i].data.copy_(t)
        else:
            result[i].data.copy_(t[:max_seq_len])
    return result

def batch_convert_len_to_mask(batch_lens, max_seq_len=-1):
    if max_seq_len == -1:
        max_seq_len = max(batch_lens)
    mask = torch.ones((len(batch_lens), max_seq_len), dtype=torch.float, requires_grad=False)
    for i, l in enumerate(batch_lens):
        mask[i, l:].fill_(0)
    return mask

class _GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

_act_map = {"none": None,
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "softmax": nn.Softmax(dim=-1),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(1/5.5),
            "prelu": nn.PReLU()}
_act_map["gelu"] = _GELU()

def map_activation_str_to_layer(act_str):
    try:
        return _act_map[act_str]
    except:
        raise NotImplementedError("Error: %s activation fuction is not supported now." % (act_str))

def anneal_fn(fn, t, T, lambda0=0.0, lambda1=1.0):
    if not fn or fn == "none":
        return lambda1
    elif fn == "logistic":
        K = 8 / T
        return float(lambda0 + (lambda1-lambda0)/(1+np.exp(-K*(t-T/2))))
    elif fn == "linear":
        return float(lambda0 + (lambda1-lambda0) * t/T)
    elif fn == "cosine":
        return float(lambda0 + (lambda1-lambda0) * (1 - math.cos(math.pi * t/T))/2)
    elif fn.startswith("cyclical"):
        R = 0.5
        t = t % T
        if t <= R * T:
            return anneal_fn(fn.split("_", 1)[1], t, R*T, lambda0, lambda1)
        else:
            return anneal_fn(fn.split("_", 1)[1], t-R*T, R*T, lambda1, lambda0)
    elif fn.startswith("anneal"):
        R = 0.5
        t = t % T
        if t <= R * T:
            return anneal_fn(fn.split("_", 1)[1], t, R*T, lambda0, lambda1)
        else:
            return lambda1
    else:
        raise NotImplementedError

def change_dropout_rate(model, dropout):
    for name, child in model.named_children():
        if isinstance(child, nn.Dropout):
            child.p = dropout
        change_dropout_rate(child, dropout)
