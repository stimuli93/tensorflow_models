import tensorflow as tf
import numpy as np


class MLP(object):
    def __init__(self, input_dim, hidden_layers, n_classes):
        """
        :param input_dim: dimension of feature vector
        :param hidden_layers: list of sizes of hidden layers
        :param n_classes: number of output classes
        """
        layers = list()
        layers.append(input_dim)
        for layer in hidden_layers:
            layers.append(layer)
        layers.append(n_classes)
        self.weights = self.initialize_weights(layers)
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.x = tf.placeholder(tf.float32, shape=[None, input_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, n_classes])
        self.reg = tf.placeholder(tf.float32)

        prev_layer = self.x
        n_layers = len(layers)
        for i in range(1, n_layers-1):
            prev_layer = tf.nn.relu(tf.matmul(prev_layer, self.weights['W'+str(i)]) + self.weights['b' + str(i)])

        i = n_layers-1
        self.y_pred = tf.nn.softmax(tf.matmul(prev_layer, self.weights['W'+str(i)]) + self.weights['b' + str(i)])

        reg_loss = 0.0
        for i in range(1, n_layers):
            reg_loss += tf.reduce_sum(self.weights['W' + str(i)] * self.weights['W' + str(i)])

        reg_loss *= self.reg
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.y_pred), reduction_indices=[1])) + reg_loss

        correct_prediction = tf.equal(tf.argmax(self.y_pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_tables())

    def initialize_weights(self, layers):
        n_layers = len(layers)
        all_weights = dict()
        for i in range(1, n_layers):
            all_weights['W' + str(i)] = tf.Variable(tf.truncated_normal(shape=[layers[i-1], layers[i]], stddev=0.1))
            all_weights['b' + str(i)] = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[layers[i]]))
        return all_weights

    def score(self, trX, trY, reg=1e-4):
        result = self.sess.run([self.cost, self.accuracy], feed_dict={self.x: trX, self.y: trY, self.reg: reg})
        return result

    def predict(self, trX):
        pred = self.sess.run([self.y_pred], feed_dict={self.x: trX})
        return np.array(pred).reshape([-1, self.n_classes])

    def train(self, trX, trY, learning_rate=1e-3, batch_size=100, reg=1e-4, n_iters=1000):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

        # initializing only un-initialized variables
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
            self.sess.run(train_step, feed_dict={self.x: trX[batch], self.y: trY[batch], self.reg: reg})
            if i % 10 == 0:
                loss = self.sess.run([self.cost],
                                     feed_dict={self.x: trX[batch], self.y: trY[batch], self.reg: reg})
                print("Loss at step %s: %s" % (i, loss))

