import json
import sys
import sklearn.preprocessing
import numpy as np
import time
import scipy.io as sio
import scipy.sparse as sparse

from dateutil import parser
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib


def timestamp_to_hour(timestamp):
    return parser.parse(timestamp).hour


def timestamp_to_weekday(timestamp):
    return parser.parse(timestamp).weekday()


def one_hot_encode(arr):
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(max(arr)+1))
    return np.array(label_binarizer.transform(arr))


def vectorize_data(data, vocabs=None):
    if vocabs is None:
        vocabs = {}
    num_instances = len(data)
    vectorized_data = sparse.csr_matrix((num_instances, 1))

    text_fields = ['targetTitle', 'targetDescription', 'targetKeywords']
    for field_name in text_fields:
        if field_name in vocabs.keys():
            field_vocab = vocabs[field_name]
            field_vector, _ = vectorize_text_field(data, field_name, vocab=field_vocab)
        else:
            field_vector, field_vocab = vectorize_text_field(data, field_name)
            vocabs[field_name] = field_vocab
        vectorized_data = add_feature(vectorized_data, field_vector)

    post_hours = [timestamp_to_hour(x['postTimestamp']) for x in data]
    post_hours = one_hot_encode(post_hours)
    vectorized_data = add_feature(vectorized_data, post_hours)

    post_weekdays = [timestamp_to_weekday(x['postTimestamp']) for x in data]
    post_weekdays = one_hot_encode(post_weekdays)
    vectorized_data = add_feature(vectorized_data, post_weekdays)

    num_paragraphs = np.array([len(x['targetParagraphs']) for x in data]).reshape((num_instances, 1))
    vectorized_data = add_feature(vectorized_data, num_paragraphs)

    has_media = np.array([len(x['postMedia']) == 0 for x in data]).reshape((num_instances, 1))
    vectorized_data = add_feature(vectorized_data, has_media)

    paragraph_len = np.array([len(' '.join(x['targetParagraphs']).split(' ')) for x in data]).reshape(
        (num_instances, 1))
    vectorized_data = add_feature(vectorized_data, paragraph_len)

    return vectorized_data, vocabs


def vectorize_text_field(data, field_name, vocab=None):
    if vocab is not None:
        vectorizer = CountVectorizer(min_df=1, binary=True, ngram_range=(1, 5), vocabulary=vocab)
    else:
        vectorizer = CountVectorizer(min_df=1, binary=True, ngram_range=(1, 5))
    if type(data[0][field_name]) == list:
        corpus = [' '.join(x[field_name]) for x in data]
    else:
        corpus = [x[field_name] for x in data]
    vectorized_field = vectorizer.fit_transform(corpus)
    return vectorized_field, vectorizer.vocabulary_


def add_feature(feature_set, feature):
    return sparse.hstack((feature_set, feature))


def check_data_label_alignment(data, labels):
    for i in range(len(data)):
        if data[i]['id'] != labels[i][0]:
            return False
    return True


def train(train_data, truth):
    regressor = RandomForestRegressor(n_estimators=30, n_jobs=-1)
    regressor = regressor.fit(train_data, truth)
    return regressor


def cross_val(regressor, train_data, truth):
    scores = cross_val_score(regressor, train_data, truth, cv=5, n_jobs=20, scoring='neg_mean_squared_error')
    print('cross val:')
    print(scores)
    return scores


def save_regressor(regressor, file_path):
    joblib.dump(regressor, file_path)


def load_regressor(file_path):
    return joblib.load(file_path)


def evaluate_regressor(regressor, test_data, truth):
    prediction = regressor.predict(test_data)
    print('MSE:')
    print(mean_squared_error(prediction, truth))

    return prediction


def load_data(data_dir):
    data = []
    for line in open(data_dir + 'instances.jsonl'):
        data.append(json.loads(line))
    return data


def load_truth(data_dir):
    labels = []
    for line in open(data_dir + 'truth.jsonl'):
        labels.append(json.loads(line))
    labels = [(x['id'], x['truthMean']) for x in labels]
    return labels


def load_and_prepare_data(data_dir):
    data = load_data(data_dir)
    truth = load_truth(data_dir)
    if not check_data_label_alignment(data, truth):
        raise Exception
    truth = [x[1] for x in truth]

    data, truth = shuffle(data, truth)

    return data, truth


# https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186#27050186
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def info(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('Function', method.__name__, 'time:', round((te - ts), 1), 'sec')
        print()
        return result
    return timed

@info
def get_data(data_dir, load_data=True, load_vocab=True):
    if load_data:
        vectorized_data = sio.mmread(data_dir + 'vectorized_data.mm').tocsr()
        truth = np.load(data_dir + 'truth.npy')
    else:
        data, truth = load_and_prepare_data(data_dir)
        if load_vocab:
            vocabs = json.load(open(data_dir+'vocabs.json', 'r'))
            vectorized_data, _ = vectorize_data(data, vocabs=vocabs)
        else:
            vectorized_data, vocabs = vectorize_data(data)
            json.dump(vocabs, open(data_dir+'vocabs.json', 'w'), cls=NumpyEncoder)
        sio.mmwrite(data_dir + 'vectorized_data.mm', vectorized_data)
        np.save(data_dir + 'truth', truth)
    return vectorized_data, truth


@info
def train_and_eval(vectorized_data, truth, holdout=0.2):
    train_test_split = int(len(truth)*holdout)
    train_truth = truth[:train_test_split]
    test_truth = truth[train_test_split:]
    print(vectorized_data.shape)
    train_data = vectorized_data[:train_test_split, :]
    test_data = vectorized_data[train_test_split:, :]
    regressor = train(train_data, train_truth)
    _ = evaluate_regressor(regressor, test_data, test_truth)
    return regressor


if __name__ == '__main__':
    if len(sys.argv) < 2:
        folder = '/home/xuri3814/data/clickbait17-validation-170616/'
    else:
        folder = sys.argv[1]

    # data_dir = '/home/neffle/data/clickbait/clickbait17-train-170331/'
    # d, t = get_data(folder, load_data=False, load_vocab=False)
    d, t = get_data(folder)
    reg = train(d, t)
    save_regressor(reg, folder + 'RandomForestRegressor_30')
    scores = cross_val(reg, d, t)
    json.dump(scores, open(folder + 'cross_val_scores.json', 'w'))
