import tensorflow as tf
import numpy as np


class LstmClassifier(object):
    def __init__(self, vocab_size, num_classes, max_seq_length, embedding_size=128, hidden_size=128,
                 batch_size=50):
        self.weights = self._initialize_weights(vocab_size, num_classes, embedding_size, hidden_size)
        self.x = tf.placeholder(tf.int32, shape=[None, max_seq_length])
        self.y = tf.placeholder(tf.float32, shape=[None, num_classes])
        self.batch_size = batch_size

        token_embedings = tf.nn.embedding_lookup(self.weights['W_vocab'], self.x)

        with tf.variable_scope("lstm") as scope:
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
            initial_state = self.cell.zero_state(batch_size, tf.float32)

            outputs = []
            states = [initial_state]
            for i in range(max_seq_length):
                if i > 0:
                    scope.reuse_variables()
                new_output, new_state = self.cell(token_embedings[:, i, :], states[-1])
                outputs.append(new_output)
                states.append(new_state)

            self.final_state = states[-1]
            self.final_output = outputs[-1]

        self.y_pred = tf.nn.softmax(tf.matmul(self.final_state[0], self.weights['W']) + self.weights['b']) + 1e-8
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.y_pred), reduction_indices=[1]))

        correct_prediction = tf.equal(tf.argmax(self.y_pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.sess = tf.InteractiveSession()

    def _initialize_weights(self, vocab_size, num_classes, embedding_size, hidden_size):
        all_weights = dict()
        all_weights['W_vocab'] = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size], stddev=0.1))
        all_weights['W'] = tf.Variable(tf.truncated_normal(shape=[hidden_size, num_classes], stddev=0.1))
        all_weights['b'] = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[num_classes]))
        return all_weights

    def score(self, X, y):
        result = self.sess.run([self.accuracy, self.cost], feed_dict={self.x: X, self.y: y})
        return result

    def predict(self, X):
        pred = self.sess.run(self.y_pred, feed_dict={self.x: X})
        return pred

    def train(self, trX, trY, learning_rate=1e-3, n_iters=100):

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        uninitialized_vars = []
        for var in tf.all_variables():
            try:
                self.sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        init_new_vars_op = tf.initialize_variables(uninitialized_vars)
        self.sess.run(init_new_vars_op)

        for i in xrange(n_iters):
            batch = np.random.randint(trX.shape[0], size=self.batch_size)
            self.sess.run(train_step, feed_dict={self.x: trX[batch], self.y: trY[batch]})
            if i % 10 == 0:
                result = self.sess.run([self.cost],
                                       feed_dict={self.x: trX[batch], self.y: trY[batch]})
                print("Loss at step %s: %s" % (i, result))

