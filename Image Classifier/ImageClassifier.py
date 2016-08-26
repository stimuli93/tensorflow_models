import tensorflow as tf
import numpy as np


class ConvolutionalClassifier(object):
    def __init__(self, n_classes=2):

        img_dims = (256, 256, 3)
        self.weights = self._initialize_weights(img_channels=img_dims[2], n_classes=n_classes)
        self.x = tf.placeholder(tf.float32, shape=[None, img_dims[0], img_dims[1], img_dims[2]], name="x_image")
        self.y = tf.placeholder(tf.float32, shape=[None, n_classes])

        h_conv1 = tf.nn.relu(self.conv2d(self.x, self.weights['W_conv1']) + self.weights['b_conv1'])
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.weights['W_conv2']) + self.weights['b_conv2'])
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, self.weights['W_conv3']) + self.weights['b_conv3'])
        h_pool3 = self.max_pool_2x2(h_conv3)

        h_pool3_flat = tf.reshape(h_pool3, [-1, 32*32*32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, self.weights['W_fc1']) + self.weights['b_fc1'])
        y_pred = tf.nn.softmax(tf.matmul(h_fc1, self.weights['W_fc2']) + self.weights['b_fc2'])
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(y_pred), reduction_indices=[1]))
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.sess = tf.Session()

        cost_summ = tf.scalar_summary("loss ", self.cost)
        accuracy_summary = tf.scalar_summary("accuracy", self.accuracy)

        # Merge all the summaries and write them out to /tmp/convClf_logs
        self.merged = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter("/tmp/convClf_logs", self.sess.graph)

        init = tf.initialize_all_variables()
        self.sess.run(init)

    def _initialize_weights(self, img_channels, n_classes):
        all_weights = dict()
        all_weights['W_conv1'] = tf.Variable(tf.truncated_normal(shape=[3, 3, img_channels, 8], stddev=0.1))
        all_weights['b_conv1'] = tf.Variable(tf.constant(0.1, shape=[8]))
        all_weights['W_conv2'] = tf.Variable(tf.truncated_normal(shape=[3, 3, 8, 16], stddev=0.1))
        all_weights['b_conv2'] = tf.Variable(tf.constant(0.1, shape=[16]))
        all_weights['W_conv3'] = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 32], stddev=0.1))
        all_weights['b_conv3'] = tf.Variable(tf.constant(0.1, shape=[32]))
        all_weights['W_fc1'] = tf.Variable(tf.truncated_normal(shape=[32*32*32, 512], stddev=0.1))
        all_weights['b_fc1'] = tf.Variable(tf.constant(0.1, shape=[512]))
        all_weights['W_fc2'] = tf.Variable(tf.truncated_normal(shape=[512, n_classes], stddev=0.1))
        all_weights['b_fc2'] = tf.Variable(tf.constant(0.1, shape=[n_classes]))

        return all_weights

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def score(self, X, y):
        result = self.sess.run([self.accuracy, self.cost], feed_dict={self.x: X, self.y: y})
        return result

    def train(self, trX, trY, learning_rate=1e-3, batch_size=10, n_iters=100, start=0):

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
            batch = np.random.randint(trX.shape[0], size=batch_size)
            self.sess.run(train_step, feed_dict={self.x: trX[batch], self.y: trY[batch]})
            if i % 10 == 0:
                result = self.sess.run([self.merged, self.accuracy], feed_dict={self.x: trX[batch], self.y: trY[batch]})
                summary_str = result[0]
                acc = result[1]
                self.writer.add_summary(summary_str, i+start)
                print("Accuracy at step %s: %s" % (i+start, acc))
