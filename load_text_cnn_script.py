import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import pandas as pd
import functools
import datetime
import name_gen as ng
import gensim

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

np.random.seed(1112)

DTYPE = 'float32'
RUN_NAME = ng.get_name()
LOG_DIR = '/home/xuri3814/data/clickbait/cnn/runs/logs/{}/'.format(RUN_NAME)
CHECKPOINT_DIR = '/home/xuri3814/data/clickbait/cnn/runs/checkpoints/{}/'.format(RUN_NAME)
DATA_DIR = '/home/xuri3814/data/clickbait/'

SEQUENCE_LENGTH = 27
OUTPUT_SIZE = 2
VOCAB_SIZE=int(3e6)
TRAIN_BATCH_SIZE=100
TEST_BATCH_SIZE=100
EMBEDDING_SIZE=300
NUM_FILTERS=64
MAX_FILTER_LENGTH=15
BETA=0.005
DROPOUT_KEEP_PROB=0.75
EMBEDDING_NAME='unknown'
LEARNING_RATE=0.05
INFO=''
DATE = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
EMBEDDING_NAME = 'googlenews300'
FILTER_SIZES = [fs for fs in range(1, MAX_FILTER_LENGTH)]
POOLING_LAYER_OUTPUT_SIZE = NUM_FILTERS * len(FILTER_SIZES)


def main():
    sequence_placeholder = tf.placeholder(tf.int32, shape=(None, SEQUENCE_LENGTH), name='sequence_placeholder')
    target_placeholder = tf.placeholder(tf.float32, shape=(None, OUTPUT_SIZE), name='target_placeholder')
    dropout_keep_prob_placeholder = tf.placeholder(tf.float32, name="dropout_keep_prob")
    embedding_placeholder = tf.placeholder(tf.float32, shape=(VOCAB_SIZE, EMBEDDING_SIZE), name='embedding_placeholder')

    with tf.device('/:cpu0'):
        embedding = tf.Variable(embedding_placeholder, trainable=False)
        embedding_lookup = tf.nn.embedding_lookup(embedding, sequence_placeholder)
        embedding_lookup_expanded = tf.expand_dims(embedding_lookup, -1)

    pooled_outputs = []
    for i, filter_size in enumerate(FILTER_SIZES):
        with tf.name_scope('convolution-maxpool-%s' % filter_size):

            # Convolution Layer
            filter_shape = [filter_size, EMBEDDING_SIZE, 1, NUM_FILTERS]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[NUM_FILTERS]), name='b')
            conv = tf.nn.conv2d(
                embedding_lookup_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='convolution')

            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

            # Max-pooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, SEQUENCE_LENGTH - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='pooling')
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    with tf.name_scope('combine_and_reshape'):
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, POOLING_LAYER_OUTPUT_SIZE])

    with tf.name_scope('dropout'):
        pooling = tf.nn.dropout(h_pool_flat, dropout_keep_prob_placeholder)


    weights = tf.get_variable(
        "output_weights",
        shape=[POOLING_LAYER_OUTPUT_SIZE, OUTPUT_SIZE],
        initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.Variable(tf.constant(0.1, shape=[OUTPUT_SIZE]), name="output_bias")

    with tf.name_scope('prediction'):
        activation = tf.matmul(pooling, weights) + bias
        softmax_out = tf.nn.softmax(activation)
        summarize_variable('activation', activation)
        summarize_variable('softmax_out', softmax_out)

    summarize_variable('bias', bias)
    summarize_variable('weights', weights)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    mse = tf.losses.mean_squared_error(target_placeholder, softmax_out)
    mse_mean = tf.reduce_mean(mse)
    log_loss = tf.losses.log_loss(labels=target_placeholder, predictions=softmax_out)
    log_loss_mean = tf.reduce_mean(log_loss)

    l2_loss = tf.nn.l2_loss(weights) + tf.nn.l2_loss(bias)
    summarize_variable('l2_loss', l2_loss)

    l2_loss_mean = tf.reduce_mean(l2_loss)
    merged_summaries = tf.summary.merge_all()
    saver = tf.train.Saver(tf.global_variables())

    with tf.name_scope('train'):
        optimizer_loss = log_loss + BETA * l2_loss
        optimizer = tf.train.AdagradOptimizer(LEARNING_RATE)
        optimize = optimizer.minimize(optimizer_loss)

    ########################################################
    ########################################################

    print(device_lib.list_local_devices())

    tokens, truth = load_data(EMBEDDING_NAME)
    num_instances, _ = tokens.shape

    train_data, test_data, train_labels, test_labels = sample_test_set(tokens, truth, 0.1)
    test_set_size = test_data.shape[0]
    num_instances -= test_set_size

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    saver.restore(sess, CHECKPOINT_DIR+'step_2000')

    train_data_batch, train_label_batch = get_random_batch(TRAIN_BATCH_SIZE, data=train_data, labels=train_labels)

    x = sess.run([softmax_out], feed_dict={ sequence_placeholder: train_data_batch,
                                            target_placeholder: train_label_batch,
                                            dropout_keep_prob_placeholder: 1.0})

    print(x)

##############################################################################
##############################################################################
##############################################################################

def summarize_variable(name_scope, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name_scope + '_summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def get_random_batch(batch_size, data, labels):
    assert batch_size < data.shape[0]
    data_batch = data.sample(batch_size)
    label_batch = labels.loc[data_batch.index]
    return data_batch, label_batch


def get_batch(data, labels, batch_size, step):
    num_samples = data.shape[0]
    assert batch_size < num_samples
    start = step * batch_size
    if start > (num_samples - 1):
        start %= num_samples
    end = start + batch_size
    if end > (num_samples - 1):
        end = num_samples - 1
    return data[start:end], labels[start:end]


def get_vocab_and_pretrained_embedding(path_to_model, binary=False):
    model = gensim.models.KeyedVectors.load_word2vec_format(path_to_model, binary=binary)
    embedding = model.syn0
    vocab = model.vocab
    return vocab, embedding


def load_data(embedding_name):
    truth = pd.read_pickle(DATA_DIR+embedding_name+'_labels.pickle')
    tokens = pd.read_pickle(DATA_DIR+embedding_name+'_indices.pickle')
    return tokens, truth


def sample_test_set(data, labels, fraction):
    """
    https://stackoverflow.com/questions/17260109/sample-two-pandas-dataframes-the-same-way
    """
    assert data.shape[0] == labels.shape[0]
    indices = np.random.binomial(1, fraction, size=data.shape[0]).astype('bool')
    train_data = data[~indices]
    test_data = data[indices]
    train_labels = labels[~indices]
    test_labels = labels[indices]
    return train_data, test_data, train_labels, test_labels


if __name__ == '__main__':
    main()