import os
import torch
import torch.nn as nn
import re
from itertools import chain
from transformers import BertTokenizer, RobertaTokenizer
from util import *

class Tokenizer:
    def __init__(self, **kw):
        pass
    
    def encode(self, sentence, return_tensors=False):
        raise NotImplementedError

    def batch_encode(self, batch_sentences, return_tensors=False, return_lengths=False, return_masks=False):
        raise NotImplementedError

    @property
    def pad_token_id(self):
        raise NotImplementedError

    def concat_sent_ids(self, sent_ids):
        raise NotImplementedError


class Word2VecTokenizer(Tokenizer):
    def __init__(self, **kw):
        super(Word2VecTokenizer, self).__init__(**kw)
        word2vec_file = kw.get("word2vec_file", "../data/model/glove/glove_wsj.txt")
        corenlp_path = kw.get("corenlp_path", "")
        corenlp_port = kw.get("corenlp_port", 0)

        self.corenlp_client = get_corenlp_client(corenlp_path=corenlp_path, corenlp_port=corenlp_port)
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
        with open(word2vec_file, "r") as f:
            line = f.readline() # word, dim
            for line in f:
                word = line.split(" ", 1)[0]
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
        
    def encode(self, sentence, return_tensors=False):
        """
        :param sentence: a string of sentence
        :return encoded_result: a list of ids or a tensor if return_tensors=True
        """
        if isinstance(sentence, str):
            tokens = chain.from_iterable(tokenize_with_corenlp(sentence, self.corenlp_client))
        else:
            tokens = sentence

        ids = [2] # <s>
        ids.extend(map(lambda t: self.word2idx.get(t, self.word2idx.get(t.lower(), 1)), tokens)) # 1 for <unk>
        ids.append(3) # </s>

        if return_tensors:
            ids = torch.tensor(ids, dtype=torch.long)
        return ids

    def batch_encode(self, batch_sentences, return_tensors=False, return_lengths=False, return_masks=False):
        """
        :param batch_sentences: a string of sentences or a list of sentences
        :return encoded_result: a list of lists of ids or a tensor if return_tensors=True and masks
        """
        if isinstance(batch_sentences, str):
            batch_tokens = tokenize_with_corenlp(sentence, self.corenlp_client)
        elif isinstance(batch_sentences, list):
            if len(batch_sentences) > 0:
                if isinstance(batch_sentences[0], str):
                    batch_tokens = [list(chain.from_iterable(tokenize_with_corenlp(sentence, self.corenlp_client))) for sentence in batch_sentences]
                else:
                    batch_to_tokens = batch_sentences
            else:
                batch_tokens = [[[]]]

        batch_ids = []
        for tokens in batch_tokens:
            batch_ids.append(self.encode(tokens, return_tensors=False))
        len_ids = [len(ids) for ids in batch_ids]
        
        if return_tensors:
            batch_ids = batch_convert_list_to_tensor(batch_ids)
            if return_masks:
                mask = batch_convert_len_to_mask(len_ids)
        else:
            if return_masks:
                mask = [[1] * l for l in len_ids]
                
        results = dict({"ids": batch_ids})
        if return_lengths:
            results["lens"] = len_ids
            if return_tensors:
                results["lens"] = torch.tensor(results["lens"])
        if return_masks:
            results["masks"] = mask
        return results

    @property
    def pad_token_id(self):
        return self.word2idx["<pad>"]

    def concat_sent_ids(self, sent_ids):
        ids = list()
        if isinstance(sent_ids[0], list):
            ids = list(chain.from_iterable(sent_ids))
        elif isinstance(sent_ids[0], torch.Tensor):
            ids = torch.cat(sent_ids, dim=-1)
        else:
            raise ValueError
        return ids


class BERTTokenizer(Tokenizer):
    def __init__(self, **kw):
        super(BERTTokenizer, self).__init__(**kw)
        corenlp_path = kw.get("corenlp_path", "")
        corenlp_port = kw.get("corenlp_port", 0)
        bert_dir = kw.get("bert_dir", "../data/pretrained_lm/bert")

        self.corenlp_client = get_corenlp_client(corenlp_path=corenlp_path, corenlp_port=corenlp_port)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=bert_dir)
    
    def encode(self, sentence, return_tensors=False):
        """
        :param sentence: a string of sentence
        :return encoded_result: a list of ids or a tensor if return_tensors=True
        """
        ids = self.tokenizer.encode(sentence)

        if return_tensors:
            ids = torch.tensor(ids)
        return ids
    
    def batch_encode(self, batch_sentences, return_tensors=False, return_lengths=False, return_masks=False):
        """
        :param batch_sentences: a string of sentences or a list of sentences
        :return encoded_result: a list of lists of ids or a tensor if return_tensors=True and masks
        """
        if isinstance(batch_sentences, str):
            batch_sentences = sentence_split_with_corenlp(batch_sentences, self.corenlp_client)

        batch_outputs = self.tokenizer.batch_encode_plus(batch_sentences, 
            return_tensors="pt" if return_tensors else None,
            return_input_lengths=return_lengths,
            return_attention_masks=return_masks)
        results = dict({"ids": batch_outputs["input_ids"]})
        if return_lengths:
            results["lens"] = batch_outputs["input_len"]
        if return_masks:
            results["masks"] = batch_outputs["attention_mask"]
        return results

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    def concat_sent_ids(self, sent_ids):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
            single sequence: [CLS] X [SEP]
            pair of sequences: [CLS] A [SEP] B [SEP]
        """
        ids = list()
        if isinstance(sent_ids[0], list):
            for i, sent_id in enumerate(sent_ids):
                if i == 0:
                    ids.extend(sent_id)
                else:
                    ids.extend(sent_id[1:])
        elif isinstance(sent_ids[0], torch.Tensor):
            for i, sent_id in enumerate(sent_ids):
                if i == 0:
                    ids.append(sent_id)
                else:
                    ids.append(sent_id[1:])
            ids = torch.cat(ids, dim=-1)
        else:
            raise ValueError
        return ids


class ROBERTATokenizer(Tokenizer):
    def __init__(self, **kw):
        super(ROBERTATokenizer, self).__init__(**kw)
        corenlp_path = kw.get("corenlp_path", "")
        corenlp_port = kw.get("corenlp_port", 0)
        roberta_dir = kw.get("roberta_dir", "../data/pretrained_lm/roberta")

        self.corenlp_client = get_corenlp_client(corenlp_path=corenlp_path, corenlp_port=corenlp_port)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", cache_dir=roberta_dir)
    
    def encode(self, sentence, return_tensors=False):
        """
        :param sentence: a string of sentence
        :return encoded_result: a list of ids or a tensor if return_tensors=True
        """
        ids = self.tokenizer.encode(sentence)

        if return_tensors:
            ids = torch.tensor(ids)
        return ids
    
    def batch_encode(self, batch_sentences, return_tensors=False, return_lengths=False, return_masks=False):
        """
        :param batch_sentences: a string of sentences or a list of sentences
        :return encoded_result: a list of lists of ids or a tensor if return_tensors=True and masks
        """
        if isinstance(batch_sentences, str):
            batch_sentences = sentence_split_with_corenlp(batch_sentences, self.corenlp_client)

        batch_outputs = self.tokenizer.batch_encode_plus(batch_sentences, 
            return_tensors="pt" if return_tensors else None,
            return_input_lengths=return_lengths,
            return_attention_masks=return_masks)
        results = dict({"ids": batch_outputs["input_ids"]})
        if return_lengths:
            results["lens"] = batch_outputs["input_len"]
        if return_masks:
            results["masks"] = batch_outputs["attention_mask"]
        return results

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    def concat_sent_ids(self, sent_ids):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A RoBERTa sequence has the following format:
            single sequence: <s> X </s>
            pair of sequences: <s> A </s></s> B </s>
        """
        ids = list()
        sep = self.tokenizer.sep_token_id
        if isinstance(sent_ids[0], list):
            for i, sent_id in enumerate(sent_ids):
                if i == 0:
                    ids.extend(sent_id)
                else:
                    ids.append(sep)
                    ids.extend(sent_id[1:])
        elif isinstance(sent_ids[0], torch.Tensor):
            for i, sent_id in enumerate(sent_ids):
                if i == 0:
                    ids.append(sent_id)
                else:
                    sent_id_ = sent_id.clone()
                    sent_id_[0].fill_(sep)
                    ids.append(sent_id_)
            ids = torch.cat(ids, dim=-1)
        else:
            raise ValueError
        return ids
