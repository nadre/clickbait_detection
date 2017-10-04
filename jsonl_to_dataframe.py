import pandas as pd
import json
import re
import time
import gensim
import argparse
import os
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer(strip_handles=True)


def info(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('Function', method.__name__, 'time:', round((te - ts)*1000, 1), 'ms')
        print()
        return result
    return timed


def strip_non_alphanum(token):
    return re.sub(r'\W+', '', token)


def tokenize(tweet_text, remove_non_alphanum=True, lowercase=False, length=27, fill_na_token='unk'):
    if lowercase:
        tweet_text = tweet_text.lower()
    tokens = tokenizer.tokenize(tweet_text)
    if remove_non_alphanum:
        tokens = [strip_non_alphanum(t) for t in tokens]
    tokens = list(filter(None, tokens))
    if len(tokens) < length:
        tokens.extend([fill_na_token] * (length - len(tokens)))
    return tokens[:length]


def get_dataframe_from_jsonl(path):
    data = []
    index = []
    for i, line in enumerate(open(path, 'r')):
        instance = json.loads(line)
        data.append(instance)
        index.append(instance['id'])
    df = pd.DataFrame(data=data, index=index)
    df.sort_index(inplace=True)
    return df


@info
def instances_to_token(path_to_instances, data_dir, file_prefix, fill_na_token='unk'):
    df_instances = get_dataframe_from_jsonl(path_to_instances)
    df_instances.to_pickle(data_dir + file_prefix + '_instances.pickle')
    df_tokens = df_instances.apply(lambda row: tokenize(row['postText'][0]), axis=1)
    df_tokens = pd.DataFrame(df_tokens.values.tolist(), df_tokens.index).add_prefix('token_')
    df_tokens.fillna(fill_na_token, inplace=True)
    df_tokens.to_pickle(data_dir + file_prefix + '_tokens.pickle')
    return df_tokens


# def token_to_index(vocab, token, unknown_token='unk'):
#     try:
#         return vocab[token].index
#     except KeyError:
#         return vocab[unknown_token].index


def token_to_index(vocab, token, unknown_token='unk'):
    try:
        return vocab[token]
    except KeyError:
        return vocab[unknown_token]


@info
def tokens_to_indices(tokens_df, data_dir, file_prefix, vocab):
    indices_df = tokens_df.applymap(lambda t: token_to_index(vocab, t))
    indices_df.to_pickle(data_dir + file_prefix + '_indices.pickle')


def get_vocab_and_pretrained_embedding(path_to_model):
    model = gensim.models.KeyedVectors.load_word2vec_format(path_to_model, binary=True)
    W = model.syn0
    print('model shape:')
    print(W.shape)
    vocab = model.vocab
    return vocab, W


def get_vocab(path_to_vocab):
    vocab = json.load(open(path_to_vocab))
    print('vocab length: {}'.format(len(vocab)))
    return vocab


def negate(x):
    return 1-x

@info
def instances_to_labels(path_to_labels, data_dir, file_prefix):
    df_truth = get_dataframe_from_jsonl(path_to_labels)
    df_truth = df_truth.ix[:, 'truthMean'].to_frame()
    df_truth['negTruthMean'] = df_truth.apply(lambda x: negate(x), axis=1)
    df_truth.to_pickle(data_dir + file_prefix + '_labels.pickle')


def main(args):
    print(args)
    path_to_vocab = args['data_dir'] + 'vocab.json'
    # vocab, _ = get_vocab_and_pretrained_embedding(path_to_model)
    vocab = get_vocab(path_to_vocab)
    df_tokens = instances_to_token(os.path.join(args['input'], 'instances.jsonl'), args['data_dir'], file_prefix='googlenews')
    tokens_to_indices(df_tokens, args['data_dir'], file_prefix='googlenews', vocab=vocab)


def gensim_model_to_vocab(args):
    path_to_model = args['data_dir'] + 'googlenews300.bin'
    vocab, _ = get_vocab_and_pretrained_embedding(path_to_model)
    vocab_dict = {k: v.index for k, v in vocab.items()}
    json.dump(vocab_dict, open(args['data_dir'] + 'vocab.json', 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-d', '--data_dir', required=True)
    args = vars(parser.parse_args())
    # gensim_model_to_vocab(args)
    main(args)
