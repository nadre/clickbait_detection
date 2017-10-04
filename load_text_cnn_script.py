import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import argparse
import json
import os


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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

np.random.seed(1112)

DTYPE = 'float32'

SEQUENCE_LENGTH = 27
OUTPUT_SIZE = 2
VOCAB_SIZE = int(3e6)
TRAIN_BATCH_SIZE = 100
TEST_BATCH_SIZE = 100
EMBEDDING_SIZE = 300
NUM_FILTERS = 64
MAX_FILTER_LENGTH = 15
BETA = 0.005
DROPOUT_KEEP_PROB = 0.75
EMBEDDING_NAME = 'unknown'
LEARNING_RATE = 0.05
INFO = ''
DATE = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
FILTER_SIZES = [fs for fs in range(1, MAX_FILTER_LENGTH)]
POOLING_LAYER_OUTPUT_SIZE = NUM_FILTERS * len(FILTER_SIZES)


def main(args):
    sequence_placeholder = tf.placeholder(tf.int32, shape=(None, SEQUENCE_LENGTH), name='sequence_placeholder')
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

    saver = tf.train.Saver(tf.global_variables())

    data = pd.read_pickle(args['data_dir']+'googlenews_indices.pickle')

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    saver.restore(sess, args['path_to_tf_model'])

    results = sess.run([softmax_out], feed_dict={sequence_placeholder: data,
                                                 dropout_keep_prob_placeholder: 1.0})[0]

    ids = list(data.index)
    with open(os.path.join(args['out_dir'], 'results.jsonl'), 'w') as out_file:
        for i, result in enumerate(results):
            line = {'id': ids[i], 'clickbaitScore': result[0]}
            out_file.write(json.dumps(line, cls=NumpyEncoder)+'\n')


##############################################################################
##############################################################################
##############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--path_to_tf_model', required=True)
    parser.add_argument('-d', '--data_dir', required=True)
    parser.add_argument('-o', '--out_dir', required=True)
    args = vars(parser.parse_args())
    main(args)
