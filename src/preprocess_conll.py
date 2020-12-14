import argparse
import os
import pandas as pd
try:
    import ujson as jsonl
except:
    import json
from functools import partial
from tokenizer import *
from dataset import Dataset
from util import str2list
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Stanford Corenlp
    parser.add_argument("--corenlp_path", type=str, default="",
                        help="StanfordCoreNLP path")
    parser.add_argument("--corenlp_port", type=int, default=9000,
                        help="port of corenlp")
    
    # Raw Data
    parser.add_argument("--json_file_path", type=str, default="", 
                        help="relations.json location")
    # Processed Data
    parser.add_argument("--processed_dir", type=str, default="",
                        help="ASER processed_dir data directory")

    # Sections to be processed
    parser.add_argument("--sections", type=str2list, default=list(),
                        help="sections")
    # Types to be processed
    parser.add_argument("--types", type=str2list, default=list(),
                        help="relation type")

    # Encoder
    parser.add_argument("--encoder", type=str, default="roberta",
                        choices=["lstm", "bert", "roberta"],
                        help="the encoder")

    # Saved dataset
    parser.add_argument("--dataset_file_path", type=str, default="",
                        help="dataset location")     
    
    # Log
    parser.add_argument("--log_path", type=str, default="./preprocess_conll.log",
                        help="log path of pdtb output")

    args = parser.parse_args()
    if args.processed_dir.endswith(os.sep):
        args.processed_dir = args.processed_dir[:-1]
    args.encoder = args.encoder.lower()
    if args.encoder == "lstm":
        tokenizer = Word2VecTokenizer(**vars(args))
    elif args.encoder == "bert":
        tokenizer = BERTTokenizer(**vars(args))
    elif args.encoder == "roberta":
        tokenizer = ROBERTATokenizer(**vars(args))
    else:
        raise NotImplementedError("Error: encoder=%s is not supported now." % (args.encoder))
    encode_func = partial(tokenizer.encode, return_tensors=True)
        
    
    # Dataset
    pdtb_dataset = Dataset(data=None, encode_func=encode_func)
    pdtb_dataset.load_json(args.json_file_path, args.processed_dir, args.sections, args.types)
    pdtb_dataset.save_pt(args.dataset_file_path)

    
        

