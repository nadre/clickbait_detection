import json
import random
import sklearn.preprocessing
import numpy as np
import scipy.sparse as sparse
from dateutil import parser
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


def timestamp_to_hour(timestamp):
    return parser.parse(timestamp).hour


def timestamp_to_weekday(timestamp):
    return parser.parse(timestamp).weekday()


def one_hot_encode(arr):
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(max(arr)+1))
    return np.array(label_binarizer.transform(arr))


def vectorize_data(data):
    num_instances = len(data)

    vectorized_data = sparse.csr_matrix((num_instances, 1))

    for i, text_field in enumerate(['targetTitle', 'targetDescription', 'targetKeywords']):
        vectorized_field, text_field_vocabulary = vectorize_text_field(data, text_field)
        vectorized_data = add_feature(vectorized_data, vectorized_field)

    vectorized_data = add_feature(vectorized_data, vectorize_text_field(data, 'targetDescription'))
    vectorized_data = add_feature(vectorized_data, vectorize_text_field(data, 'targetKeywords'))

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

    return vectorized_data


def vectorize_text_field(data, field_name):
    vectorizer = CountVectorizer(min_df=1, binary=True, ngram_range=(1, 5))
    if type(data[0][field_name]) == list:
        corpus = [' '.join(x[field_name]) for x in data]
    else:
        corpus = [x[field_name] for x in data]
    vectorized_field = vectorizer.fit_transform(corpus)
    return vectorized_field, vectorizer.vocabulary


def add_feature(feature_set, feature):
    return sparse.hstack((feature_set, feature))


def check_data_label_alignment(data, labels):
    for i in range(len(data)):
        if data[i]['id'] != labels[i]['id']:
            return False
    return True


def balance_data(data, labels):
    """
    Assuming there are more negative then positive samples!
    """
    num_samples = len(labels)
    num_pos_samples = sum([x for x in labels if x])
    num_neg_samples = num_samples - num_pos_samples

    imbalance = num_neg_samples - num_pos_samples
    indexes_of_pos_samples = [i for i, x in enumerate(labels) if x]

    for n in range(imbalance):
        idx = random.choice(indexes_of_pos_samples)
        data.append(data[idx])
        labels.append(labels[idx])

    return data, labels


def train_and_eval():
    data = []
    for line in open('/home/xuri3814/data/clickbait17-train-170331/instances.jsonl'):
        data.append(json.loads(line))

    labels = []
    for line in open('/home/xuri3814/data/clickbait17-train-170331/truth.jsonl'):
        labels.append(json.loads(line))

    if not check_data_label_alignment(data, labels):
        raise Exception

    labels = [x['truthClass'] == 'clickbait' for x in labels]

    data, labels = balance_data(data, labels)

    data, labels = shuffle(data, labels)

    vectorized_data, vocabs = vectorize_data(data).tocsr()
    vectorized_data = vectorized_data.tocsr()
    train_test_split = 2394
    train_data = vectorized_data[:train_test_split, :]
    test_data = vectorized_data[train_test_split:, :]
    train_labels = labels[:train_test_split]
    test_labels = labels[train_test_split:]

    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(train_data, train_labels)

    scores = cross_val_score(clf, train_data, train_labels)
    print('cross val:')
    print(scores)

    predicted_labels = clf.predict(test_data)
    print('roc auc:')
    print(roc_auc_score(predicted_labels, test_labels))

    print('confusion:')
    print(confusion_matrix(predicted_labels, test_labels))

    print('report:')
    target_names = ['clickbait', 'no-clickbait']
    print(classification_report(predicted_labels, test_labels, target_names=target_names))

    print('false positive target titles:')
    for i, predicted_clickbait in enumerate(predicted_labels):
        if predicted_clickbait and not test_labels[i]:
            print(data[train_test_split+i]['targetTitle'])


if __name__ == '__main__':
    train_and_eval()
