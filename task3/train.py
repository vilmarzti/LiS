import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score


# Network Parameters
n_input     = 93
n_hidden_1  = 256
n_hidden_2  = 128
n_classes   = 5

# Params
display_size = 10
epochs = 800

#Load data
data = pd.read_hdf('train.h5', "train")
test = pd.read_hdf('test.h5', "test")

zerosColumns = (data!=0).any(axis=0)
data = data.loc[:,zerosColumns]


weights = {
    'h1'    : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2'    : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out'   : tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'h1'    : tf.Variable(tf.random_normal([n_hidden_1])),
    'h2'    : tf.Variable(tf.random_normal([n_hidden_2])),
    'out'   : tf.Variable(tf.random_normal([n_classes]))
}

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

def MLP(x, weights, biases):
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['h2'])
    layer2 = tf.nn.relu(layer2)

    layer_out = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
    return layer_out


mlp = MLP(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mlp,
                                                              labels=y))

optimizer = tf.train.AdamOptimizer().minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    avg_cost = 0
    step = 0

    y_col = data.get('y').tolist()[:3500]
    y_col = tf.one_hot(y_col, n_classes).eval()
    x_col = data.drop('y', 1).values[:3500]

    y_test = data.get('y').tolist()[3500:]
    x_test = data.drop('y', 1).values[3500:]
    y_hot = tf.one_hot(y_col, n_classes).eval()
    prediction = tf.argmax(mlp, 1)

    for index in range(epochs):
        _, c = sess.run([optimizer, cost], feed_dict={x: x_col,
                                                      y: y_col})
        avg_cost += c/float(display_size)

        if index % display_size == 0:
            acc = accuracy_score(prediction.eval({x: x_test}), y_test)

            print("Step {} - Cost: {}".format(index, avg_cost))
            print("sklearn acc: {}".format(acc))
            print("\n")

            avg_cost = 0

    acc = accuracy_score(prediction.eval({x: x_test}), y_test)

    correct_predictions = tf.equal(tf.argmax(mlp, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

    print("accuracy", accuracy.eval({x: x_col, y: y_col}))
    print("sklearn acc: {}".format(acc))
