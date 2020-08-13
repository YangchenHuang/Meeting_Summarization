import os
from collections import Counter
import glob
import re
import argparse

import utils
import clustering


path_to_stopwords = 'resources/stopwords.txt'
path_to_filler_words = 'resources/filler_words.txt'
stopwords = utils.load_stopwords(path_to_stopwords)
filler_words = utils.load_filler_words(path_to_filler_words)


def process_corpus(type='story'):
    if type == 'story':
        path = args.transcript_path+'*'+'transcript.txt'
    elif type == 'txt':
        path = args.txt_path+'*'+'.txt'
    corpus = {}
    for text in glob.glob(path):
        f = open(text, "r").read()
        if type == 'story':
            id = re.sub(args.transcript_path, '', text)
            id = re.sub('.transcript.txt', '', id)
        elif type == 'txt':
            id = re.sub(args.txt_path, '', text)
            id = re.sub('.txt', '', id)
        raw = re.split('\. |\? |\! |\; ', f)
        utterances = []
        for i, utt in enumerate(raw):
            utt = utt.lower()
            utt = utils.clean_utterance(utt, filler_words=filler_words)
            utt = re.sub(' +', ' ', utt)
            utt = re.sub('\n', '', utt)
            utt = utt.strip()

            if utt != '' and utt != '.' and utt != ' ':
                utterances.append((i, utt))
        corpus[id] = utterances
    return corpus


def process_summary():
    summary = {}
    for text in glob.glob(args.summary_path + '*' + 'abssumm.txt'):
        f = open(text, "r").read()
        id = re.sub(args.summary_path, '', text)
        id = re.sub('.abssumm.txt', '', id)
        raw = re.split('\. |\? |\! |\; ', f)
        utterances = []
        for utt in raw:
            utt = utt.lower()
            utt = re.sub(' +', ' ', utt)
            utt = re.sub('\n', '', utt)
            utt = utt.strip()

            if utt != '' and utt != '.' and utt != ' ':
                utterances.append(utt)
        summary[id] = utterances
    return summary


def segmentation_story(corpus, summary, args, ids=None, type='all'):
    if ids == None:
        ids = summary.keys()
    for id in ids:
        print(id)

        utterances_indexed = corpus[id]

        summ = summary[id]

        # remove stopwords and short sentences
        utterances_processed = []
        for utterance_indexed in utterances_indexed:
            index, utt = utterance_indexed
            utt_cleaned = utils.clean_text(utt, stopwords)
            if len(utt_cleaned) >= args.min_words:
                utterances_processed.append((index, ' '.join(utt_cleaned)))
        print(len(utterances_processed), 'utterances')

        # apply clustering algorithm
        membership = clustering.cluster_utterances(utterances_processed, args)
        c = dict(Counter(membership))
        comm_labels = [k for k, v in c.items()]

        path_to_community = args.story_path + type + '/'
        path_to_index = args.index_path + type + '/'
        if not os.path.exists(path_to_community):
            os.makedirs(path_to_community)
        if not os.path.exists(path_to_index):
            os.makedirs(path_to_index)

        for i, label in enumerate(comm_labels):
            output_index = []
            with open(path_to_community + id + '.' + str(i) + '.story', 'w+') as story_file:
                for my_label in [sent[0] for j, sent in enumerate(utterances_processed) if membership[j] == label]:
                    selected = [elt for elt in utterances_indexed if elt[0] == my_label][0]
                    output_index.append(selected[0])
                    to_write =selected[1]
                    story_file.write(to_write + '. \n')
                for sent in summ:
                    story_file.write('@highlight {}\n'.format(sent))
                story_file.write('\n')
            with open(path_to_index + id + '.' + str(i) + '.index', 'w+') as index_file:
                for idx in output_index:
                    index_file.write(str(idx) + ' ')


def segmentation_txt(corpus, args):
    for id in corpus.keys():
        print(id)

        utterances_indexed = corpus[id]

        summary_txt = 'This is a sample summary'

        # remove stopwords and short sentences
        utterances_processed = []
        for utterance_indexed in utterances_indexed:
            index, utt = utterance_indexed
            utt_cleaned = utils.clean_text(utt, stopwords)
            if len(utt_cleaned) >= args.min_words:
                utterances_processed.append((index, ' '.join(utt_cleaned)))
        print(len(utterances_processed), 'utterances')

        # apply clustering algorithm
        membership = clustering.cluster_utterances(utterances_processed, args)
        c = dict(Counter(membership))
        comm_labels = [k for k, v in c.items()]

        path_to_community = args.txt_out_path + 'test/'
        if not os.path.exists(path_to_community):
            os.makedirs(path_to_community)
        path_to_index = args.index_path + 'test/'
        if not os.path.exists(path_to_index):
            os.makedirs(path_to_index)
        for i, label in enumerate(comm_labels):
            output_index = []
            with open(path_to_community + id + '.' + str(i) + '.story', 'w+') as story_file:
                for my_label in [sent[0] for j, sent in enumerate(utterances_processed) if membership[j] == label]:
                    selected = [elt for elt in utterances_indexed if elt[0] == my_label][0]
                    output_index.append(selected[0])
                    to_write = selected[1]
                    story_file.write(to_write + '. \n')
                story_file.write('@highlight {}\n'.format(summary_txt))
                story_file.write('\n')
            with open(path_to_index + id + '.' + str(i) + '.index', 'w+') as index_file:
                for idx in output_index:
                    index_file.write(str(idx) + ' ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='story', choices=['story', 'txt'], help='output format')
    parser.add_argument('-split', type=bool, default=True, help='whether need train test split')
    parser.add_argument('-seed', type=int, default=2020, help='random seed for split')
    parser.add_argument("-transcript_path", default='../raw_data/transcript/')
    parser.add_argument("-summary_path", default='../raw_data/summary/')
    parser.add_argument("-txt_path", default='../txt_data/')
    parser.add_argument("-index_path", default='../ext_data/index/')
    parser.add_argument("-story_path", default='../ext_data/story/')
    parser.add_argument("-txt_out_path", default='../ext_data/text/')
    parser.add_argument('-algorithm', type=str, default='ec', choices=['ec', 'kmeans'],
                        help='clustering algorithm: equal size cluster or kmeans')
    parser.add_argument('-n_gram', type=str, default='1, 1',
                        help='The lower and upper boundary of the range of n-values for different '
                             'n-grams to be extracted.')
    parser.add_argument('-lsa_num', type=int, default=30, help='feature dimension for lsa')
    parser.add_argument('-sent_num', type=int, default=10, help='sentence num for each cluster')
    parser.add_argument('-min_words', type=int, default=3, help='sentence with less than min_words non-stopwords will '
                                                                'be dropped')

    args = parser.parse_args()

    if args.mode == 'story':

        corpus = process_corpus()
        summary = process_summary()
        if args.split:
            train, valid, test = utils.train_test_split(summary, args)
            dmap = {'train': train, 'valid': valid, 'test': test}
            for corpus_type in dmap.keys():
                segmentation_story(corpus, summary, args, dmap[corpus_type], corpus_type)
        else:
            segmentation_story(corpus, summary, args)
    elif args.mode == 'txt':
        corpus = process_corpus('txt')
        segmentation_txt(corpus, args)





