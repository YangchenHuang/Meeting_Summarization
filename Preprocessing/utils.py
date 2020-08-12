import re
import string
import nltk
import random
from random import shuffle


def load_stopwords(path):
    stopwords = set([])
    with open(path, 'r+') as f:
        temp = f.read().splitlines()
        for line in temp:
            if not re.search('^#', line) and len(line.strip()) > 0:
                stopwords.add(line.strip().lower())
    return stopwords


def load_filler_words(path):
    with open(path, 'r+') as f:
        filler = f.read().splitlines()

    return filler


def clean_utterance(utterance, filler_words):
    utt = utterance
    # fix common ASR error
    utt = re.sub("'Kay", 'Okay', utt)
    utt = re.sub("'kay", 'Okay', utt)
    utt = re.sub('"Okay"', 'Okay', utt)
    utt = re.sub("'cause", 'cause', utt)
    utt = re.sub("'Cause", 'cause', utt)
    utt = re.sub('"cause"', 'cause', utt)
    utt = re.sub('"\'em"', 'them', utt)
    utt = re.sub('"\'til"', 'until', utt)
    utt = re.sub('"\'s"', 's', utt)
    utt = re.sub('[.\n]', ' ', utt)
    utt = re.sub('h. t. m. l.', 'html', utt)
    utt = re.sub(r"(\w)\_ (\w)\_ (\w)\_", r"\1\2\3", utt)
    utt = re.sub(r"(\w)\_ (\w)\_", r"\1\2", utt)
    utt = re.sub(r"(\w)\_", r"\1", utt)

    # replace consecutive terms with only one
    utt = re.sub('\\b(\\w+)\\s+\\1\\b', '\\1', utt)
    utt = re.sub('(\\b.+?\\b)\\1\\b', '\\1', utt)

    utt = re.sub(' +', ' ', utt)
    utt = utt.strip()

    # remove filler words
    utt = ' ' + utt + ' '
    for filler_word in filler_words:
        utt = re.sub(' ' + filler_word + ' ', ' ', utt)
        utt = re.sub(' ' + filler_word + ', ', ' ', utt)
        utt = re.sub(' ' + filler_word + '. ', '.', utt)

    utt = re.sub(' +', ' ', utt)
    utt = utt.strip()

    return utt


def clean_text(text, stopwords):
    text = text.lower()
    text = re.sub(' +', ' ', text)
    text = text.strip()
    tokens = text.split(' ')

    # remove punctuation
    tokens = [t for t in tokens if t not in string.punctuation]

    # remove stopwords
    tokens = [token for token in tokens if token.lower() not in stopwords]

    # apply Porter's stemmer
    stemmer = nltk.stem.PorterStemmer()
    # apply Porter's stemmer
    tokens_stemmed = list()
    for token in tokens:
        tokens_stemmed.append(stemmer.stem(token))
    tokens = tokens_stemmed

    return tokens


def train_test_split(summary, args):
    train_files, valid_files, test_files = [], [], []
    dataset = list(summary.keys())
    size = len(dataset)
    random.seed(args.seed)
    shuffle(dataset)
    for i, f in enumerate(dataset):
        if i < size * 0.1:
            test_files.append(f)
        elif i < size * 0.2:
            valid_files.append(f)
        else:
            train_files.append(f)
    return train_files, valid_files, test_files