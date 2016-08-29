import tensorflow as tf
import numpy as np
import os


class ConvolutionalClassifier(object):
    def __init__(self, img_dims, n_classes=2):
        """
        :param img_dims: a tuple of (width, heigh, channel) of input image.
        Image width/height expected to be multiples of 8
        :param n_classes: number of output categories
        """
        self.weights = self._initialize_weights(img_dims=img_dims, n_classes=n_classes)
        self.x = tf.placeholder(tf.float32, shape=[None, img_dims[0], img_dims[1], img_dims[2]], name="x_image")
        self.y = tf.placeholder(tf.float32, shape=[None, n_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        h_conv1 = tf.nn.relu(self.conv2d(self.x, self.weights['W_conv1']) + self.weights['b_conv1'])
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.weights['W_conv2']) + self.weights['b_conv2'])
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, self.weights['W_conv3']) + self.weights['b_conv3'])
        h_pool3 = self.max_pool_2x2(h_conv3)

        h_pool3_flat = tf.reshape(h_pool3, [-1, (img_dims[0]//8)*(img_dims[1]//8)*16])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, self.weights['W_fc1']) + self.weights['b_fc1'])
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, self.weights['W_fc2']) + self.weights['b_fc2'])
        h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)
        self.y_pred = tf.nn.softmax(tf.matmul(h_fc2_drop, self.weights['W_fc3']) + self.weights['b_fc3']) + 1e-8

        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.y_pred), reduction_indices=[1]))

        correct_prediction = tf.equal(tf.argmax(self.y_pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.sess = tf.InteractiveSession()

        cost_summ = tf.scalar_summary("loss ", self.cost)
        accuracy_summary = tf.scalar_summary("accuracy", self.accuracy)

        # Merge all the summaries and write them out to /tmp/convClf_logs
        self.merged = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter("/tmp/convClf_logs", self.sess.graph)

        ckpt_dir = "./ckpt_dir"
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Call this after declaring all tf.Variables.
        self.saver = tf.train.Saver()

        init = tf.initialize_all_variables()
        self.sess.run(init)

    def _initialize_weights(self, img_dims, n_classes):
        all_weights = dict()
        all_weights['W_conv1'] = tf.Variable(tf.truncated_normal(shape=[3, 3, img_dims[2], 4], stddev=0.1))
        all_weights['b_conv1'] = tf.Variable(tf.constant(0.1, shape=[4]))
        all_weights['W_conv2'] = tf.Variable(tf.truncated_normal(shape=[3, 3, 4, 8], stddev=0.1))
        all_weights['b_conv2'] = tf.Variable(tf.constant(0.1, shape=[8]))
        all_weights['W_conv3'] = tf.Variable(tf.truncated_normal(shape=[3, 3, 8, 16], stddev=0.1))
        all_weights['b_conv3'] = tf.Variable(tf.constant(0.1, shape=[16]))
        all_weights['W_fc1'] = tf.Variable(tf.truncated_normal(shape=[(img_dims[0]//8)*(img_dims[1]//8)*16, 256],
                                                               stddev=0.1))
        all_weights['b_fc1'] = tf.Variable(tf.constant(0.1, shape=[256]))
        all_weights['W_fc2'] = tf.Variable(tf.truncated_normal(shape=[256, 64], stddev=0.1))
        all_weights['b_fc2'] = tf.Variable(tf.constant(0.1, shape=[64]))
        all_weights['W_fc3'] = tf.Variable(tf.truncated_normal(shape=[64, n_classes], stddev=0.1))
        all_weights['b_fc3'] = tf.Variable(tf.constant(0.1, shape=[n_classes]))

        return all_weights

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def score(self, X, y):
        result = self.sess.run([self.accuracy, self.cost], feed_dict={self.x: X, self.y: y, self.keep_prob: 1.0})
        return result

    def predict(self, X):
        pred = self.sess.run(self.y_pred, feed_dict={self.x:X, self.keep_prob: 1.0})
        return pred

    def train(self, trX, trY, learning_rate=1e-3, batch_size=10, n_iters=100, keep_prob=0.5):

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        uninitialized_vars = []
        for var in tf.all_variables():
            try:
                self.sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        init_new_vars_op = tf.initialize_variables(uninitialized_vars)
        self.sess.run(init_new_vars_op)

        ckpt_dir = "./ckpt_dir"
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # restore all variables

        start = self.global_step.eval()  # get last global_step
        for i in xrange(n_iters):
            batch = np.random.randint(trX.shape[0], size=batch_size)
            self.sess.run(train_step, feed_dict={self.x: trX[batch], self.y: trY[batch], self.keep_prob: keep_prob})
            if i % 10 == 0:
                result = self.sess.run([self.merged, self.cost],
                                       feed_dict={self.x: trX[batch], self.y: trY[batch], self.keep_prob: 1.0})
                summary_str = result[0]
                loss = result[1]
                self.writer.add_summary(summary_str, i+start)
                print("Loss at step %s: %s" % (i+start, loss))
                if i % 100 == 0:
                    self.global_step.assign(i + start).eval()  # set and update(eval) global_step with index, i
                    self.saver.save(self.sess, "./ckpt_dir/model.ckpt", global_step=self.global_step)
