import numpy as np
import tensorflow as tf


class CharRNN(object):
    def __init__(self, max_seq_length, vocab_size, hidden_size=32, embedding_size=16, batch_size=32):

        self.weights = self.initialize_weights(vocab_size, hidden_size, embedding_size)
        self.x = tf.placeholder(tf.int32, shape=[batch_size, max_seq_length])
        self.y = tf.placeholder(tf.int32, shape=[batch_size, max_seq_length])
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        token_embeddings = tf.nn.embedding_lookup(self.weights['W_vocab'], self.x)

        self.cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
        state = self.cell.zero_state(batch_size, tf.float32)
        outputs = []
        for i in range(max_seq_length):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            output, state = self.cell(token_embeddings[:, i, :], state)
            outputs.append(output)
        outputs = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
        logits = tf.matmul(outputs, self.weights['W1']) + self.weights['b1']
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.y, [-1])],
            [tf.ones([batch_size * max_seq_length], dtype=tf.float32)])
        self.cost = tf.reduce_sum(loss) / batch_size
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

    def initialize_weights(self, vocab_size, hidden_size, embedding_size):
        all_weights = dict()
        all_weights['W_vocab'] = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
        all_weights['W1'] = tf.Variable(tf.truncated_normal([hidden_size, vocab_size]))
        all_weights['b1'] = tf.Variable(tf.constant(0.1, shape=[vocab_size]))
        return all_weights

    def train(self, X, learning_rate=1e-3, n_iters=1000):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

        # initializing only un-initialized variables to prevent trained variables assigned again with random weights
        uninitialized_vars = []
        for var in tf.all_variables():
            try:
                self.sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        init_new_vars_op = tf.initialize_variables(uninitialized_vars)
        self.sess.run(init_new_vars_op)
        for iteration in range(n_iters):
            batches = np.random.randint(len(X)-self.max_seq_length, size=self.batch_size)
            trX = []
            trY = []
            for i in batches:
                trX.append(X[i:i+self.max_seq_length])
                trY.append(X[i+1:i+1+self.max_seq_length])
            trX = np.array(trX)
            trY = np.array(trY)
            self.sess.run([train_step], feed_dict={self.x: trX, self.y: trY})
            if iteration % 10 == 0:
                cost = self.sess.run([self.cost], feed_dict={self.x: trX, self.y: trY})
                print 'Loss at iteration %s is: %s' % (iteration, cost)
