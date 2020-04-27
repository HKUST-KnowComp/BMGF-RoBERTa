import pickle
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import re
import math
from collections import defaultdict
from copy import copy
from itertools import chain
from tqdm import tqdm
from utils import *

class WordMapper:
    def __init__(self, **kw):
        word2vec_file = kw.get("word2vec_file", "../data/model/glove/glove_wsj.txt")
        self.word_to_add = kw.get("word_to_add", dict())

        self.word2idx = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
        with open(word2vec_file, "r") as f:
            line = f.readline() # token, dim
            for line in f:
                word = line.split(" ", 1)[0]
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
    
    def convert_word_to_id(self, word):
        if word in self.word_to_add:
            return self.word_to_add[word]
        else:
            return self.word2idx.get(word, 1)

    def get_num_words(self):
        return max(len(self.word2idx),  max(self.word_to_add.values())+1 if len(self.word_to_add) else 0)


class Dataset(data.Dataset):
    rel_map_14 = defaultdict(lambda: -1, 
        {"Temporal.Asynchronous.Precedence": 0, # "Precedence"
        "Temporal.Asynchronous.Succession": 1, # "Succession"
        "Temporal.Synchrony": 2, # "Synchrony"
        "Contingency.Cause.Reason": 3, # "Reason"
        "Contingency.Cause.Result": 4, # "Result"
        "Contingency.Condition.Hypothetical": 5, # "Condition"
        "Comparison.Contrast": 6, # "Contrast"
        "Comparison.Contrast.Juxtaposition": 6, # "Contrast"
        "Comparison.Contrast.Opposition": 6, # "Contrast"
        "Comparison.Concession": 7, # "Concession"
        "Comparison.Concession.Contra-expectation": 7, # "Concession"
        "Comparison.Concession.Expectation": 7, # "Concession"
        "Expansion.Conjunction": 8, # "Conjunction"
        "Expansion.Instantiation": 9, # "Instantiation"
        "Expansion.Restatement": 10, # "Restatement"
        "Expansion.Restatement.Equivalence": 10, # "Restatement"
        "Expansion.Restatement.Generalization": 10, # "Restatement"
        "Expansion.Restatement.Specification": 10, # "Restatement"
        "Expansion.Alternative": 11, # "Alternative"
        "Expansion.Alternative.Conjunctive": 11, # "Alternative"
        "Expansion.Alternative.Chosen alternative": 12, # ChosenAlternative
        "Expansion.Exception": 13}) # "Exception"

    rel_map_12 = defaultdict(lambda: -1, 
        {"Comparison.Concession": 0,
        "Comparison.Contrast": 1,
        "Contingency.Cause.Reason": 2,
        "Contingency.Cause.Result": 3,
        "Expansion": 4,
        "Expansion.Alternative.Chosen alternative": 5,
        "Expansion.Conjunction": 6,
        "Expansion.Instantiation": 7,
        "Expansion.Restatement": 8,
        "Temporal.Asynchronous.Precedence": 9,
        "Temporal.Asynchronous.Succession": 10,
        "Temporal.Synchrony": 11})

    rel_map_11 = defaultdict(lambda: -1, 
        {# "Comparison",
        "Comparison.Concession": 0,
        "Comparison.Concession.Contra-expectation": 0,
        "Comparison.Concession.Expectation": 0,
        "Comparison.Contrast": 1,
        "Comparison.Contrast.Juxtaposition": 1,
        "Comparison.Contrast.Opposition": 1,
        # "Comparison.Pragmatic concession",
        # "Comparison.Pragmatic contrast",
        # "Contingency",
        "Contingency.Cause": 2,
        "Contingency.Cause.Reason": 2,
        "Contingency.Cause.Result": 2,
        # "Contingency.Condition.Hypothetical",
        "Contingency.Pragmatic cause.Justification": 3,
        # "Contingency.Pragmatic condition.Relevance",
        # "Expansion",
        "Expansion.Alternative": 4,
        "Expansion.Alternative.Chosen alternative": 4,
        "Expansion.Alternative.Conjunctive": 4,
        "Expansion.Conjunction": 5,
        # "Expansion.Exception",
        "Expansion.Instantiation": 6,
        "Expansion.List": 7,
        "Expansion.Restatement": 8,
        "Expansion.Restatement.Equivalence": 8,
        "Expansion.Restatement.Generalization": 8,
        "Expansion.Restatement.Specification": 8,
        # "Temporal",
        "Temporal.Asynchronous.Precedence": 9,
        "Temporal.Asynchronous.Succession": 9,
        "Temporal.Synchrony": 10})
    # Counter({'Comparison.Concession': 216,
    #      'Comparison.Contrast': 1915,
    #      'Comparison.Pragmatic concession': 1,
    #      'Comparison.Pragmatic contrast': 4,
    #      'Contingency.Cause': 3833,
    #      'Contingency.Condition': 1,
    #      'Contingency.Pragmatic cause': 78,
    #      'Contingency.Pragmatic condition': 1,
    #      'Expansion.Alternative': 171,
    #      'Expansion.Conjunction': 3355,
    #      'Expansion.Exception': 2,
    #      'Expansion.Instantiation': 1332,
    #      'Expansion.List': 360,
    #      'Expansion.Restatement': 2945,
    #      'Temporal.Asynchronous': 662,
    #      'Temporal.Synchrony': 245})

    rel_map_4 = defaultdict(lambda: -1, 
        {"Comparison": 0,
        "Comparison.Concession": 0,
        "Comparison.Concession.Contra-expectation": 0,
        "Comparison.Concession.Expectation": 0,
        "Comparison.Contrast": 0,
        "Comparison.Contrast.Juxtaposition": 0,
        "Comparison.Contrast.Opposition": 0,
        "Comparison.Pragmatic concession": 0,
        "Comparison.Pragmatic contrast": 0,
        "Contingency": 1,
        "Contingency.Cause": 1,
        "Contingency.Cause.Reason": 1,
        "Contingency.Cause.Result": 1,
        "Contingency.Condition.Hypothetical": 1,
        "Contingency.Pragmatic cause.Justification": 1,
        "Contingency.Pragmatic condition.Relevance": 1,
        "Expansion": 2,
        "Expansion.Alternative": 2,
        "Expansion.Alternative.Chosen alternative": 2,
        "Expansion.Alternative.Conjunctive": 2,
        "Expansion.Conjunction": 2,
        "Expansion.Exception": 2,
        "Expansion.Instantiation": 2,
        "Expansion.List": 2,
        "Expansion.Restatement": 2,
        "Expansion.Restatement.Equivalence": 2,
        "Expansion.Restatement.Generalization": 2,
        "Expansion.Restatement.Specification": 2,
        "Temporal": 3,
        "Temporal.Asynchronous.Precedence": 3,
        "Temporal.Asynchronous.Succession": 3,
        "Temporal.Synchrony": 3})
    # Counter({'Comparison': 2293,
    #      'Contingency': 3917,
    #      'Expansion': 8256,
    #      'Temporal': 909})

    def __init__(self, data=None, encode_func=None):
        super(Dataset, self).__init__()
        if data is not None:
            self.data = data
        else:
            self.data = list()
        if encode_func is not None:
            self.encode_func = encode_func
        else:
            self.encode_func = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod 
    def batchify(batch, rel_map, min_arg=3, max_arg=512, pad_id=0):
        assert isinstance(rel_map, defaultdict)
        len_rel = max(rel_map.values())+1
        arg_size = list()
        for x in batch:
            arg_size = list(x["arg1"].size())[1:]
            break

        valid_batch, prefered_relation = list(), list()
        for i, x in enumerate(batch):
            for r in x["rel_strs"]:
                idx = rel_map[r]
                if idx != -1:
                    valid_batch.append(x)
                    prefered_relation.append(idx)
                    break
        bsz = len(valid_batch)

        arg1_lens = [x["arg1"].size(0) for x in valid_batch]
        arg2_lens = [x["arg2"].size(0) for x in valid_batch]
        max_arg1 = min(max(min_arg, max(arg1_lens)), max_arg)
        max_arg2 = min(max(min_arg, max(arg2_lens)), max_arg)
        arg1 = torch.empty([bsz, max_arg1] + arg_size, dtype=torch.long).fill_(pad_id)
        arg1_mask = batch_convert_len_to_mask(arg1_lens, max_arg1)
        arg2 = torch.empty([bsz, max_arg2] + arg_size, dtype=torch.long).fill_(pad_id)
        arg2_mask = batch_convert_len_to_mask(arg2_lens, max_arg2)
            
        _id, relation = list(), list()
        for i, x in enumerate(valid_batch):
            _id.append(x["id"])
            l1 = arg1_lens[i]
            if l1 < max_arg1:
                arg1[i, :l1].data.copy_(x["arg1"])
            elif l1 == max_arg1:
                arg1[i].data.copy_(x["arg1"])
            else:
                arg1[i].data.copy_(x["arg1"][:max_arg1])
            l2 = arg2_lens[i]
            if l2 < max_arg2:
                arg2[i, :l2].data.copy_(x["arg2"])
            elif l2 == max_arg2:
                arg2[i].data.copy_(x["arg2"])
            else:
                arg2[i].data.copy_(x["arg2"][:max_arg2])
            rel = torch.zeros((len_rel,), dtype=torch.float, requires_grad=False)
            for r in x["rel_strs"]:
                idx = rel_map[r]
                if idx != -1:
                    rel[idx] = 1
            relation.append(rel)
        relation = torch.cat(relation, dim=0).view(bsz, len_rel)
        prefered_relation = torch.tensor(prefered_relation, dtype=torch.long)
        
        return _id, arg1, arg1_mask, arg2, arg2_mask, relation, prefered_relation
    
    def load_csv(self, csv_file_path, processed_dir, sections=None, types=None):
        self.data = list()

        df = pd.read_csv(csv_file_path, usecols=[
            "Relation", "Section", "FileNumber", "SentenceNumber", 
            "ConnHeadSemClass1", "ConnHeadSemClass2", 
            "Conn2SemClass1", "Conn2SemClass2",
            "Arg1_RawText", "Arg2_RawText", "FullRawText"])
        if sections:
            df = df[df["Section"].isin(set(sections))]
        if types:
            df = df[df["Relation"].isin(set(types))]
        df.fillna("", inplace=True)

        parsed_result = list()
        for idx, row in tqdm(df.iterrows()):
            rel_strs = list()
            if row[0] == "EntRel":
                rel_strs.append(row[0])
            else:
                if row[4]:
                    rel_strs.append(row[4])
                if row[5]:
                    rel_strs.append(row[5])
                if row[6]:
                    rel_strs.append(row[6])
                if row[7]:
                    rel_strs.append(row[7])

            x = {"id": "%d_%s_wsj_%02d%02d" % (idx, row[0], row[1], row[2]),
                "rel_strs": rel_strs,
                "arg1": self.encode_func(row[8]),
                "arg2": self.encode_func(row[9])}
            self.data.append(x)

    def load_json(self, json_file_path, processed_dir, sections=None, types=None):
        self.data = list()

        df = pd.read_json(json_file_path, lines=True)

        if sections:
            df = df[df["Section"].isin(set(sections))]
        if types:
            df = df[df["Type"].isin(set(types))]
        df.fillna("", inplace=True)

        curr_filename = ""
        parsed_result = list()
        for idx, row in tqdm(df.iterrows()):
            rel_strs = row[5]

            x = {"id": "%d_%s_%s" % (idx, row[6], row[3]),
                "rel_strs": rel_strs,
                "arg1": self.encode_func(row[0]["RawText"]),
                "arg2": self.encode_func(row[1]["RawText"])}
            self.data.append(x)

    def load_pickle(self, pkl_file_path):
        with open(pkl_file_path, "rb") as f:
            data = pickle.load(f)
            self.data = data
            return self

    def save_pickle(self, pkl_file_path):
        with open(pkl_file_path, "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)


class Sampler(data.Sampler):
    _type_map = {
        int: np.int32,
        float: np.float32}

    def __init__(self, dataset, group_by, batch_size, shuffle, drop_last):
        super(Sampler, self).__init__(dataset)
        if isinstance(group_by, str):
            group_by = [group_by]
        for attr in group_by:
            setattr(self, attr, list())
        self.data_size = len(dataset.data)
        for x in dataset.data:
            for attr in group_by:
                value = x[attr]
                if hasattr(value, "__len__"):
                    getattr(self, attr).append(len(value))
                else:
                    getattr(self, attr).append(value)
        self.order = copy(group_by)
        self.order.append("rand")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def make_array(self):
        self.rand = np.random.rand(self.data_size).astype(np.float32)
        if self.data_size == 0:
            types = [np.float32] * len(self.order)
        else:
            types = [type(getattr(self, attr)[0]) for attr in self.order]
            types = [Sampler._type_map.get(t, t) for t in types]
        dtype = list(zip(self.order, types))
        array = np.array(
            list(zip(*[getattr(self, attr) for attr in self.order])),
            dtype=dtype)
        return array

    def __iter__(self):
        array = self.make_array()
        indices = np.argsort(array, axis=0, order=self.order)
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        batch_idx = 0
        while batch_idx < len(batches)-1:
            yield batches[batch_idx]
            batch_idx += 1
        if len(batches) > 0 and (len(batches[batch_idx]) == self.batch_size or not self.drop_last):
            yield batches[batch_idx]

    def __len__(self):
        if self.drop_last:
            return math.floor(self.data_size/self.batch_size)
        else:
            return math.ceil(self.data_size/self.batch_size)
