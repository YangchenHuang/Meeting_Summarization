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


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    words = sum(sentences, [])

    return _get_ngrams(n, words)


def load_json(p):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if tokens[0] == '@highlight':
            flag = True
            tgt.append([])
            continue
        if (flag):
            tgt[-1].extend(tokens)
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt


def tokenize(args):
    stories_dir = args.raw_path
    tokenized_stories_dir = args.token_path
    if not os.path.exists(tokenized_stories_dir):
        os.makedirs(tokenized_stories_dir)
    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if not s.endswith('story'):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


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

        print(tgt_subtokens_str)
        print(tgt_subtoken_idxs)
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
            real_name = json_f.split('\\')[-1]
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
            real_name = json_f.split('\\')[-1]
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
    token_path = args.token_path
    if not os.path.exists(args.json_path):
        os.makedirs(args.json_path)
    train_files = glob.glob(token_path + '/train' + '*.story.json')
    valid_files = glob.glob(token_path + '/valid' + '*.story.json')
    test_files = glob.glob(token_path + '/test' + '*.story.json')

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
    source, tgt = load_json(f)
    id = re.sub(args.token_path + '\\\\' + corpus_type + '.', '', f)
    id = re.sub('.story.json', '', id)
    return {'src': source, 'tgt': tgt, 'id': id}
