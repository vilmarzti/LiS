import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score


# Network Parameters
n_input   = 100
n_classes = 2

# Params
display_size = 3
epochs = 10000


#Load data
data = pd.read_hdf('train.h5', "train")
test = pd.read_hdf('test.h5', "test")


class MLP(object):
    def __init__(self, n_input, layer_weights):
        self.n_layers = len(layer_weights)
        self.n_input  = n_input
        self.n_output = layer_weights[-1]
        self.layer_weights = layer_weights
        self.weights = []
        self.biases  = []

        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.int32, [None])

        with tf.variable_scope("layer-weights"):
            prev_size = 0

            for index, size in enumerate(self.layer_weights):
                if index == 0:
                    weight = tf.get_variable("weight-{}".format(str(index)),
                                             shape=(n_input, size))
                else:
                    weight = tf.get_variable("weig-{}".format(str(index)),
                                             shape=(prev_size, size))

                bias = tf.get_variable("bias-{}".format(str(index)),
                                       shape=(size))

                self.weights.append(weight)
                self.biases.append(bias)
                prev_size = size

        with tf.variable_scope("applied-layers"):
            prev_layer = 0
            for x in range(self.n_layers):
                if x == 0:
                    layer = self.layer(tf.nn.relu, self.x, self.weights[x], self.biases[x])
                elif x == self.n_layers - 1:
                    layer = self.layer(tf.identity, prev_layer, self.weights[x], self.biases[x])
                else:
                    layer = self.layer(tf.nn.relu, prev_layer, self.weights[x], self.biases[x])
                prev_layer = layer

            self.out_layer = prev_layer

        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out_layer,
                                                                       labels=self.y)

        self.cost = tf.reduce_mean(self.cross_entropy)
        self.prediction = tf.argmax(self.out_layer, 1)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

    def layer(self, function, inp,  weights, bias):
        y = tf.matmul(inp, weights) + bias
        return function(y)


with tf.Session() as sess:
    mlp = MLP(n_input, [2048, 1024, 512, 256, 128, 64, 32, n_classes])

    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter('./logs', sess.graph)
    step = 0

    y_col = data.get('y').tolist()[:3500]
    y_col = [1 if y==0 else 0 for y in y_col]

    x_col = data.drop('y', 1).values[:3500]

    y_test = data.get('y').tolist()[3500:]
    y_col = [1 if y==0 else 0 for y in y_test]

    x_test = data.drop('y', 1).values[3500:]

    for index in range(epochs):
        _ = sess.run(mlp.optimizer, feed_dict={mlp.x: x_col,
                                               mlp.y: y_col})

        if index % display_size == 0:
            y_pred = sess.run(mlp.prediction, feed_dict={mlp.x: x_test})
            acc = accuracy_score(y_pred, y_test)

            print("Step {}".format(index))
            print("sklearn acc: {}".format(acc))
            print("\n")

    x_pred = sess.run(mlp.prediction, feed_dict={mlp.x:x_test})
    acc = accuracy_score(x_pred,  y_test)
    print("sklearn acc: {}".format(acc))
