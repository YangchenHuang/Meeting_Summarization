import gc
import glob
import json
import os
import re
from os.path import join as pjoin
import numpy as np

import torch
from transformers import BertTokenizer


REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


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

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def load_story(p):
    src = []
    tgt = []
    flag = False
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


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def weight_calculation(doc_sent_list, abstract_sent_list):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    candidates_1 = set.union(*map(set, evaluated_1grams))
    candidates_2 = set.union(*map(set, evaluated_2grams))
    rouge_1 = cal_rouge(candidates_1, reference_1grams)['r']
    rouge_2 = cal_rouge(candidates_2, reference_2grams)['r']

    rouge_score = rouge_1 + rouge_2

    return rouge_score


def sentence_onehot(doc_sent_list, abstract_sent_list):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    rouge_scores=[]
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    for idx, s in enumerate(doc_sent_list):
        candidates_1 = [evaluated_1grams[idx]]
        candidates_1 = set.union(*map(set, candidates_1))
        candidates_2 = [evaluated_2grams[idx]]
        candidates_2 = set.union(*map(set, candidates_2))
        rouge_1 = cal_rouge(candidates_1, reference_1grams)['r']
        rouge_2 = cal_rouge(candidates_2, reference_2grams)['r']
        rouge_score = rouge_1 + rouge_2
        rouge_scores.append(rouge_score)
    idx = np.argmax(rouge_scores)
    one_hot = [0]*len(rouge_scores)
    one_hot[idx] = 1
    return one_hot


def sentence_dist(doc_sent_list, abstract_sent_list):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    rouge_scores=[]
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    for idx, s in enumerate(doc_sent_list):
        candidates_1 = [evaluated_1grams[idx]]
        candidates_1 = set.union(*map(set, candidates_1))
        candidates_2 = [evaluated_2grams[idx]]
        candidates_2 = set.union(*map(set, candidates_2))
        rouge_1 = cal_rouge(candidates_1, reference_1grams)['r']
        rouge_2 = cal_rouge(candidates_2, reference_2grams)['r']
        rouge_score = rouge_1 + rouge_2
        rouge_scores.append(rouge_score)
    sum_score = sum(rouge_scores)
    if sum_score!=0:
        rouge_scores = [rouge_score/sum_score for rouge_score in rouge_scores]
    return rouge_scores


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def score_preprocess(self, src, tgt, scores):

        if len(src) == 0:
            return None

        original_src_txt = [' '.join(s) for s in src]

        labels = scores

        idxs = [i for i, s in enumerate(src) if len(s) > self.args.min_src_ntokens_per_sent]

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        labels = [labels[i] for i in idxs]

        src_txt = [' '.join(sent) for sent in src]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = labels[:len(cls_ids)]

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt


def format_to_bert(args):
    if not os.path.exists(args.bert_path):
        os.makedirs(args.bert_path)
    datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.json_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('\\')[-1]
            a_lst.append((json_f, args, pjoin(args.bert_path, real_name.replace('json', 'bert.pt'))))
        for a in a_lst:
            _format_to_bert(a)
        print("Complete to build %s bert files" % corpus_type)


def _format_to_bert(params):
    json_file, args, save_file = params
    bert = BertData(args)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt, index = d['src'], d['tgt'], d['index']
        if args.mode == 'sent_dist':
            rouge_scores = sentence_dist(source, tgt)
        elif args.oracle_mode == 'sent_one':
            rouge_scores = sentence_onehot(source, tgt)
        # b_data = bert.preprocess(source, tgt, oracle_ids)
        weight = weight_calculation(source, tgt)
        b_data = bert.score_preprocess(source, tgt, rouge_scores)
        if b_data is None:
            continue
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt, 'weight': weight, 'index': index}
        datasets.append(b_data_dict)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(args):
    if not os.path.exists(args.json_path):
        os.makedirs(args.json_path)
    index_path = args.index_path
    data = [[] for i in range(3)]
    for i, corpus_type in enumerate(['train', 'valid', 'test']):
        path = args.story_path + corpus_type + '/'
        files = glob.glob(pjoin(path, '*.story'))
        if len(files) == 0:
            continue
        for f in files:
            id = re.sub(args.story_path + corpus_type+ '\\\\', '', f)
            id = re.sub('.story', '', id)
            data[i].append((id, f))
    for i, corpus_type in enumerate(['train', 'valid', 'test']):
        dataset = []
        current_id = None
        count = 0
        if len(data[i]) == 0:
            continue
        data[i].sort(key=lambda x: x[0])
        for id, f in data[i]:
            d = _format_to_lines(f, id, corpus_type, index_path)
            id = re.sub('\.\d+', '', id)
            if current_id != None and id != current_id:
                pt_file = "{:s}/{:s}.{:s}.json".format(args.json_path, corpus_type, current_id)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    count += 1
                    dataset = []
            # print(current_id, corpus_type)
            current_id = id
            dataset.append(d)
        pt_file = "{:s}/{:s}.{:s}.json".format(args.json_path, corpus_type, current_id)
        with open(pt_file, 'w') as save:
            # save.write('\n'.join(dataset))
            save.write(json.dumps(dataset))
            count += 1
            dataset = []
        print("Complete to format %s files as one json file" % corpus_type)


def _format_to_lines(f, id, corpus_type, index_path):
    source, tgt = load_story(f)
    path = index_path + corpus_type + '/'
    with open(path + id +'.index', 'r+') as f:
        index = [int(x) for x in f.read().split()]

    return {'src': source, 'tgt': tgt, 'index': index}
