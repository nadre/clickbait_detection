import pandas as pd
import json
import re
import time
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer(strip_handles=True)


def info(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('Function', method.__name__, 'time:', round((te -ts)*1000,1), 'ms')
        print()
        return result
    return timed


def normalize_token(token):
    return re.sub(r'\W+', '', token)


def tokenize(tweet_text):
    tokens = tokenizer.tokenize(tweet_text)
    tokens = [normalize_token(t) for t in tokens]
    return list(filter(None, tokens))


def get_dataframe_from_jsonl(path):
    data = []
    index = []
    for i, line in enumerate(open(path, 'r')):
        instance = json.loads(line)
        data.append(instance)
        index.append(instance['id'])
    return pd.DataFrame(data=data, index=index)


@info
def instances_to_token(path_to_instances):
    df_instances = get_dataframe_from_jsonl(path_to_instances)
    df_instances.to_pickle(data_dir + 'all_instances.pickle')
    df_tokens = df_instances.apply(lambda row: tokenize(row['postText'][0]), axis=1)
    df_tokens = pd.DataFrame(df_tokens.values.tolist(), df_tokens.index).add_prefix('token_')
    df_tokens.fillna('UNK', inplace=True)
    df_tokens.to_pickle(data_dir + 'all_tokens.pickle')


@info
def instances_to_truth(path_to_truth):
    df_truth = get_dataframe_from_jsonl(path_to_truth)['truthMean']
    df_truth.to_pickle(data_dir + 'all_truth.pickle')

if __name__ == '__main__':
    data_dir = '/home/xuri3814/data/clickbait/'
    path_to_instances = data_dir + 'all_instances.jsonl'
    path_to_labels = data_dir + 'all_truth.jsonl'
    instances_to_token(path_to_instances)
    instances_to_truth(path_to_labels)
