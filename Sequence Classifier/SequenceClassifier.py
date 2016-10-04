import tensorflow as tf
import numpy as np
import os


class LstmClassifier(object):
    def __init__(self, vocab_size, num_classes, max_seq_length, embedding_size=128, hidden_size=128, num_layers=2,
                 batch_size=50, ckpt_dir="./ckpt_dir", summary_dir="/tmp/lstmClf_logs"):
        """
        :param vocab_size: vocabulary size
        :param num_classes: no. of output classes
        :param max_seq_length: number of words in a single training example
        :param embedding_size: size of embedding
        :param hidden_size: size of hidden unit of lstm
        :param num_layers: number of layers in lstm
        :param batch_size: no. of training samples to be trained in one iteration
        :param ckpt_dir: directory in which model checkpoints to be stored
        :param summary_dir: directory used as logdir for tensoboard visualization
        """
        self.weights = self._initialize_weights(vocab_size, num_classes, embedding_size, hidden_size)
        self.x = tf.placeholder(tf.int32, shape=[None, max_seq_length], name='X')
        self.y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')

        # keep_prob is less than 1.0 only during training
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.ckpt_dir = ckpt_dir
        self.summary_dir = summary_dir
        self.num_layers = num_layers

        with tf.variable_scope("embeddings"):
            token_embedings = tf.nn.embedding_lookup(self.weights['W_vocab'], self.x)
            token_embedings = tf.nn.dropout(token_embedings, keep_prob=self.keep_prob)

        with tf.variable_scope("lstm") as scope:
            cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True),
                                                 output_keep_prob=self.keep_prob)
            self.cell = tf.nn.rnn_cell.MultiRNNCell([cell]*num_layers, state_is_tuple=True)
            initial_state = self.cell.zero_state(batch_size, tf.float32)

            state = initial_state
            for i in range(max_seq_length):
                if i > 0:
                    scope.reuse_variables()
                _, state = self.cell(token_embedings[:, i, :], state)

            self.final_state = state

        with tf.variable_scope("loss"):
            # adding 1e-8 to avoid log(0)
            self.y_pred = tf.nn.softmax(tf.matmul(self.final_state[num_layers-1][0], self.weights['W']) +
                                        self.weights['b']) + 1e-8
            self.cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.y_pred), reduction_indices=[1]))

        with tf.variable_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.y_pred, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess = tf.Session()

        cost_summ = tf.scalar_summary("loss ", self.cost)
        accuracy_summary = tf.scalar_summary("accuracy", self.accuracy)

        # Merge all the summaries and write them out to summary_dir
        self.merged = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter(self.summary_dir, self.sess.graph)

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Call this after declaring all tf.Variables.
        self.saver = tf.train.Saver()

        init = tf.initialize_all_variables()
        self.sess.run(init)

    def _initialize_weights(self, vocab_size, num_classes, embedding_size, hidden_size):
        all_weights = dict()
        all_weights['W_vocab'] = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size], stddev=0.1),
                                             name="W_vocab")
        all_weights['W'] = tf.Variable(tf.truncated_normal(shape=[hidden_size, num_classes], stddev=0.1), name="W")
        all_weights['b'] = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[num_classes]), name="b")
        return all_weights

    def score(self, X, y):
        """
        :param X: training features
        :param y: training labels
        :return: accuracy & cost on the given data
        """
        result = np.array([0.0, 0.0], dtype=np.float64)
        denominator = 0
        lim = X.shape[0] - self.batch_size + 1
        for i in range(0, lim, self.batch_size):
            itr_result = self.sess.run([self.accuracy, self.cost], feed_dict={self.x: X[i:i+self.batch_size],
                                                                              self.y: y[i:i+self.batch_size],
                                                                              self.keep_prob: 1.0})
            result += np.array(itr_result)
            denominator += 1

        if denominator == 0:
            return result
        return np.array(result)/denominator

    def predict_prob(self, X):
        # Restore model before predicting labels for given data
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # restore all variables

        pred = np.zeros(dtype=np.float64, shape=[X.shape[0], self.num_classes])
        lim = X.shape[0] - self.batch_size + 1
        for i in range(0, lim, self.batch_size):
            itr_pred = self.sess.run(self.y_pred, feed_dict={self.x: X[i:i+self.batch_size], self.keep_prob: 1.0})
            pred[i:i+self.batch_size] = np.array(itr_pred).reshape([self.batch_size, self.num_classes])

        return pred

    def train(self, trX, trY, learning_rate=1e-3, n_iters=100, keep_prob=0.8):

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

        # restore model
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # restore all variables

        start = self.sess.run(self.global_step)  # get last global_step

        for i in xrange(n_iters):
            batch = np.random.randint(trX.shape[0], size=self.batch_size)
            self.sess.run(train_step, feed_dict={self.x: trX[batch], self.y: trY[batch], self.keep_prob: keep_prob})
            if i % 10 == 0:
                result = self.sess.run([self.merged, self.cost],
                                       feed_dict={self.x: trX[batch], self.y: trY[batch], self.keep_prob: 1.0})
                summary_str = result[0]
                loss = result[1]
                self.writer.add_summary(summary_str, i + start)
                print("Loss at step %s: %s" % (i + start, loss))
                if i%100 == 0:
                    self.sess.run(self.global_step.assign(start))
                    self.saver.save(self.sess, self.ckpt_dir+"/model.ckpt", global_step=self.global_step)
        self.sess.run(self.global_step.assign(n_iters + start))
        self.saver.save(self.sess, self.ckpt_dir + "/model.ckpt", global_step=self.global_step)
