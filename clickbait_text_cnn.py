import tensorflow as tf
import numpy as np
import pandas as pd
import functools
import datetime
import os
import name_gen as ng
import gensim
np.random.seed(1991)

DTYPE = 'float32'
RUN_NAME = ng.get_name()
LOG_DIR = '/home/xuri3814/data/clickbait/cnn/runs/logs/{}/'.format(RUN_NAME)
CHECKPOINT_DIR = '/home/xuri3814/data/clickbait/cnn/runs/checkpoints/{}/'.format(RUN_NAME)
DATA_DIR = '/home/xuri3814/data/clickbait/'


def lazy_property(function):
    """
    http://danijar.com/structuring-your-tensorflow-models/
    http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
    paper: http://arxiv.org/abs/1408.5882
    :param function:
    :return:
    """
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            # print(function.__name__)
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class Model:
    def __init__(self, name, sequence_length, output_size, vocab_size=int(3e6), train_batch_size=80, test_batch_size=80,
                 embedding_size=300, num_filters=32, max_filter_length=15, beta=0.00001, dropout_keep_prob=0.5,
                 embedding_name='unknown'):

        self.name = name
        self.date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        self.embedding_name = embedding_name

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = [fs for fs in range(1, max_filter_length)]
        self.num_filters = num_filters
        self.pooling_layer_output_size = self.num_filters * len(self.filter_sizes)
        self.sequence_length = sequence_length
        self.dropout_keep_prob = dropout_keep_prob
        self.beta = beta

        self.lowest_mse = 999.0
        self.best_step = 0

        self._sequence_placeholder = tf.placeholder(tf.int32, shape=(None, self.sequence_length))
        self._target_placeholder = tf.placeholder(tf.float32, shape=(None, self.output_size))
        self._dropout_keep_prob_placeholder = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.embedding_placeholder
        self.embedding
        self.embedding_lookup
        self.convolution_and_max_pooling
        self.prediction
        self.optimize
        self.merged_summaries
        self.output_weights_and_bias
        self.global_step
        self.l2_loss

    def get_info(self):
        info = ''
        for attr, value in self.__dict__.items():
            if not attr.startswith('_') and not callable(value):
                info += '{}: {}\n'.format(attr, value)
        return info

    def save_info(self, info_dir, fname):
        os.makedirs(info_dir, exist_ok=True)
        with open(info_dir + fname, 'w') as text_file:
            print(self.get_info(), file=text_file)

    @lazy_property
    def prediction(self):
        pooling = self.convolution_and_max_pooling

        with tf.name_scope('prediction'):
            weights, bias = self.output_weights_and_bias
            activation = tf.matmul(pooling, weights) + bias
            softmax_out = tf.nn.softmax(activation)
            summarize_variable('activation', activation)
            summarize_variable('softmax_out', softmax_out)

        summarize_variable('bias', bias)
        summarize_variable('weights', weights)
        summarize_variable('l2_loss', self.l2_loss)
        return softmax_out

    @lazy_property
    def convolution_and_max_pooling(self):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('convolution-maxpool-%s' % filter_size):

                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='b')
                conv = tf.nn.conv2d(
                    self.embedding_lookup,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='convolution')

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pooling')
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        with tf.name_scope('combine_and_reshape'):
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, self.pooling_layer_output_size])

        with tf.name_scope('dropout'):
            h_pool_flat = tf.nn.dropout(h_pool_flat, self._dropout_keep_prob_placeholder)

        return h_pool_flat

    @lazy_property
    def output_weights_and_bias(self):
        weights = tf.get_variable(
            "output_weights",
            shape=[self.pooling_layer_output_size, self.output_size],
            initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.Variable(tf.constant(0.1, shape=[self.output_size]), name="output_bias")
        return weights, bias

    @lazy_property
    def embedding_lookup(self):
        with tf.device('/:cpu0'):
            embedding_lookup = tf.nn.embedding_lookup(self.embedding, self._sequence_placeholder)
            embedding_lookup_expanded = tf.expand_dims(embedding_lookup, -1)
            return embedding_lookup_expanded

    @lazy_property
    def embedding_placeholder(self):
        """
        https://stackoverflow.com/questions/35394103/initializing-tensorflow-variable-with-an-array-larger-than-2gb
        :return:
        """
        return tf.placeholder(tf.float32, shape=(self.vocab_size, self.embedding_size))

    @lazy_property
    def embedding(self):
        embedding = tf.Variable(self.embedding_placeholder)
        return embedding

    @lazy_property
    def optimize(self):
        with tf.name_scope('train'):
            loss = self.mse + self.beta * self.l2_loss
            optimizer = tf.train.AdamOptimizer()
            return optimizer.minimize(loss, global_step=self.global_step)

    @lazy_property
    def global_step(self):
        return tf.Variable(0, name="global_step", trainable=False)

    @lazy_property
    def mse(self):
        return tf.losses.mean_squared_error(self._target_placeholder, self.prediction)

    @lazy_property
    def mse_mean(self):
        return tf.reduce_mean(self.mse)

    @lazy_property
    def log_loss(self):
        return tf.losses.log_loss(labels=self._target_placeholder, predictions=self.prediction)

    @lazy_property
    def log_loss_mean(self):
        return tf.reduce_mean(self.log_loss)

    @lazy_property
    def l2_loss(self):
        weights, bias = self.output_weights_and_bias
        return tf.nn.l2_loss(weights) + tf.nn.l2_loss(bias)

    @lazy_property
    def l2_loss_mean(self):
        return tf.reduce_mean(self.l2_loss)

    @lazy_property
    def merged_summaries(self):
        return tf.summary.merge_all()

    @lazy_property
    def checkpoint(self):
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        return tf.train.Saver(tf.global_variables())


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


def get_batch(x, y, batch_size, step, num_samples):
    assert batch_size < num_samples

    start = step * batch_size
    if start > (num_samples - 1):
        start %= num_samples
    end = start + batch_size
    if end > (num_samples - 1):
        end = num_samples - 1

    return x[start:end], y[start:end]


def get_vocab_and_pretrained_embedding(path_to_model, binary=False):
    model = gensim.models.KeyedVectors.load_word2vec_format(path_to_model, binary=binary)
    W = model.syn0
    vocab = model.vocab
    return vocab, W


def load_data():
    truth = pd.read_pickle(DATA_DIR+'glove.6B.200d_labels.pickle')
    tokens = pd.read_pickle(DATA_DIR+'glove.6B.200d_indices.pickle')
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


def evaluate_test_set(model, sess, test_data, test_labels, train_step, summary_writer):
    test_set_size = test_data.shape[0]
    num_test_steps = int(test_set_size/model.test_batch_size) + 1
    errors = {
        'mse': [],
        'log_loss': [],
        'l2_loss': []
    }
    for test_step in range(num_test_steps):
        test_data_batch, test_label_batch = get_batch(test_data, test_labels, model.test_batch_size,
                                                      test_step, test_set_size)
        mse, log_loss, l2_loss, summary = sess.run([
            model.mse_mean,
            model.log_loss_mean,
            model.l2_loss_mean,
            model.merged_summaries
        ], feed_dict={model._sequence_placeholder: test_data_batch,
                      model._target_placeholder: test_label_batch,
                      model._dropout_keep_prob_placeholder: 1.0})

        errors['mse'].append(mse)
        errors['log_loss'].append(log_loss)
        errors['l2_loss'].append(l2_loss)

        print('\n\n'
              'Train Step: {}\n'
              'Test Step: {}\n'
              'MSE {:6.10f}\n'
              'Log Loss {:6.10f}\n'
              'L2 Loss {:6.10f}\n'
              .format(train_step, test_step, mse, log_loss, l2_loss))

        summary_writer.add_summary(summary, train_step + test_step)

    error_description_df = pd.DataFrame.from_dict(errors).describe()
    summary = tf.Summary()
    print(RUN_NAME+'#'*(80 - len(RUN_NAME)))
    for key in error_description_df.keys():
        for measurement in ['mean', 'std']:
            print('{} {} : {:6.10f}'.format(key, measurement, error_description_df[key][measurement]))
            tag = 'test_{}_{}'.format(key, measurement)
            summary.value.add(tag=tag, simple_value=error_description_df[key][measurement])
    print('#'*80)
    summary_writer.add_summary(summary, train_step)

    mse = error_description_df['mse']['mean']

    if mse < model.lowest_mse:
        model.lowest_mse = mse
        model.best_step = train_step
        model.save_info(LOG_DIR, RUN_NAME + '.txt')
        model.checkpoint.save(sess, CHECKPOINT_DIR, global_step=train_step)


def main():
    tokens, truth = load_data()
    num_instances, sequence_length = tokens.shape
    _, output_size = truth.shape

    train_data, test_data, train_labels, test_labels = sample_test_set(tokens, truth, 0.1)
    test_set_size = test_data.shape[0]
    num_instances -= test_set_size

    embedding_name = 'glove.6B.200d.w2v.bin'

    print('loading embedding...')
    vocab, embedding = get_vocab_and_pretrained_embedding(DATA_DIR + embedding_name, binary=True)
    print('...done.')

    vocab_size, embedding_size = embedding.shape

    model = Model(RUN_NAME, sequence_length, output_size,
                  vocab_size=vocab_size, embedding_size=embedding_size, embedding_name=embedding_name)

    print(model.get_info())
    model.save_info(LOG_DIR, RUN_NAME + '.txt')

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    sess.run(tf.global_variables_initializer(), feed_dict={model.embedding_placeholder: embedding})

    print('started running: ' + RUN_NAME)
    for train_step in range(100000):
        train_data_batch, train_label_batch = get_batch(train_data, train_labels, model.train_batch_size,
                                                        train_step, num_instances)

        sess.run([model.optimize], feed_dict={model._sequence_placeholder: train_data_batch,
                                              model._target_placeholder: train_label_batch,
                                              model._dropout_keep_prob_placeholder: model.dropout_keep_prob})

        if train_step != 0 and train_step % 500 == 0:
            evaluate_test_set(model, sess, test_data, test_labels, train_step, summary_writer)


if __name__ == '__main__':
    main()
