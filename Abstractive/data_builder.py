import gc
import glob
import json
import re
import os
import subprocess
from os.path import join as pjoin

import torch
from transformers import BertTokenizer
from transformers import LongformerTokenizer

from others.utils import clean


def load_story(p):
    src = []
    tgt = []
    with open(p, 'r+') as f:
        for sent in f.readlines():
            tokens = sent.split()
            if len(tokens) > 0:
                if tokens[0] == '@highlight':
                    tokens.pop(0)
                    tgt.append(tokens)
                else:
                    src.append(tokens)

    src = [clean(' '.join(sent)).split() for sent in src]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return src, tgt


class LongformerData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.sep_token = '</s>'
        self.pad_token = '<pad>'
        self.tgt_bos = '<bos>'
        self.tgt_eos = '<eos>'
        self.tgt_sent_split = '<q>'
        self.pad_vid = self.tokenizer.convert_tokens_to_ids(self.pad_token)

    def preprocess(self, src, tgt):
        special_tokens_dict = {'bos_token': '<bos>', 'eos_token': '<eos>', 'additional_special_tokens': ['<q>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        src = src[:self.args.max_src_nsents]

        src_txt = [' '.join(sent) for sent in src]
        text = '{} '.format(self.sep_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

        tgt_subtokens_str = '<bos> ' + ' <q> '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt))) for tt in tgt]) + ' <q>' + ' <eos>'
        tgt_subtoken = tgt_subtokens_str.split()
        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        return src_subtoken_idxs, tgt_subtoken_idxs, src_txt, tgt_txt


def format_to_longformer(args):
    if not os.path.exists(args.long_path):
        os.makedirs(args.long_path)
    json_path = args.json_path
    datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(json_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, pjoin(args.long_path, real_name.replace('json', 'pt'))))
        for a in a_lst:
            _format_to_longformer(a)
        print("Complete to build %s longformer files" % corpus_type)


def _format_to_longformer(params):
    corpus_type, json_file, args, save_file = params

    longformer = LongformerData(args)

    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt, id = d['src'], d['tgt'], d['id']
        b_data = longformer.preprocess(source, tgt)

        if b_data is None:
            continue
        src_subtoken_idxs, tgt_subtoken_idxs, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs, 'src_txt': src_txt, "tgt_txt": tgt_txt,
                       "id": id}
        datasets.append(b_data_dict)

    torch.save(datasets, save_file)
    gc.collect()


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt):

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        src = src[:self.args.max_src_nsents]

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)


        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt))) for tt in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()
        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, tgt_subtoken_idxs,  src_txt, tgt_txt


def format_to_bert(args):
    if not os.path.exists(args.bert_path):
        os.makedirs(args.bert_path)
    json_path = args.json_path
    datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(json_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, pjoin(args.bert_path, real_name.replace('json', 'pt'))))
        for a in a_lst:
            _format_to_bert(a)
        print("Complete to build %s bert files" % corpus_type)


def _format_to_bert(params):
    corpus_type, json_file, args, save_file = params
    bert = BertData(args)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt, id = d['src'], d['tgt'], d['id']

        b_data = bert.preprocess(source, tgt)

        if b_data is None:
            continue
        src_subtoken_idxs, tgt_subtoken_idxs, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs, 'src_txt': src_txt, "tgt_txt": tgt_txt,
                       "id": id}
        datasets.append(b_data_dict)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(args):
    # stanfordnlp.download('en')
    story_path = args.story_path
    if not os.path.exists(args.json_path):
        os.makedirs(args.json_path)
    train_files = glob.glob(story_path + '/train' + '*.story')
    valid_files = glob.glob(story_path + '/valid' + '*.story')
    test_files = glob.glob(story_path + '/test' + '*.story')

    corpus = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        dataset = []
        p_ct = 0
        for f in corpus[corpus_type]:
            d = _format_to_lines(f, corpus_type, args)
            dataset.append(d)
            if len(dataset) > args.shard_size:
                pt_file = "{:s}./{:s}.{:d}.json".format(args.json_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        if len(dataset) > 0:
            pt_file = "{:s}./{:s}.{:d}.json".format(args.json_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []
        print("Complete to format %s files as one json file" % corpus_type)


def _format_to_lines(f, corpus_type, args):
    source, tgt = load_story(f)
    id = re.sub(args.story_path + corpus_type + '.', '', f)
    id = re.sub('.story', '', id)
    return {'src': source, 'tgt': tgt, 'id': id}
