import numpy as np
import tensorflow as tf


def neural_net_text_input(text_vector_len, name):
    """
    Return a Tensor for a batch of text input
    : text_vector_len: int, Length of text vector
    : return: placeholder for text vector input.
    """
    return tf.placeholder(tf.float32, [None, text_vector_len], name=name)


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: int, number of classes
    : return: placeholder for label input.
    """
    return tf.placeholder(tf.float32, [None, n_classes], 'y')


def neural_net_weight_tensor(word_weights):
    """
    Return a Tensor for the weight matrix
    :param word_weights: List<Double>
    :return: A matrix tensor with variable
    """
    d = len(word_weights)
    matrix = []

    for idx, weight in enumerate(word_weights):
        diag = tf.Variable([weight], trainable=True)
        diag_before = [0.0]*idx
        diag_before = tf.constant(diag_before)
        diag_after = [0.0]*(d-1-idx)
        diag_after = tf.constant(diag_after)
        row = tf.concat(values=[diag_before, diag, diag_after], axis=0)
        matrix.append(row)

    matrix = [[row] for row in matrix]
    matrix = tf.concat(values=matrix, axis=0)

    return matrix


def similarity_matrix(weight_matrix, input_host, input_guest):
    """
    给一组句子host，给一组句子guest，算guest和host的相似度
    weight_matrix: tensor with variable
    input_host: placeholder
    input_guest: placeholder
    :return: a matrix tensor with variable and placeholder
    """
    output_host = tf.matmul(input_host, weight_matrix)
    output_guest = tf.matmul(input_guest, weight_matrix)
    similarity = tf.matmul(output_guest, tf.transpose(output_host))

    similarity = tf.identity(similarity, name='similarity')

    return similarity


def get_probabilities_cost(similarity, y):
    """
    计算每个样本在各个类别的概率

    similarity: tensor with variable and placeholder
    y: placeholder
    :return: tensor with variable and placeholder
    """
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=similarity, labels=y))
    return cost


def get_optimizer_single(cost):
    """
    host中每个类别只有一个句子; 完成一次梯度下降
    cost: tensor with variable and placeholder
    :return: optimizer
    """
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    return optimizer


def get_accuracy(similarity, y):
    """
    计算正确率
    :param similarity: tensor with variable and placeholder
    :param y: placeholder
    :return: tensor with variable and placeholder
    """
    correct_pred = tf.equal(tf.argmax(similarity, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),
                              name='accuracy')
    return accuracy


def train_batch(epochs):

    host = [[1, 0, 1, 0], [1, 1, 1, 1]]
    guest = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]]
    labels = [[1, 0], [1, 0], [0, 1], [0, 1]]
    init_weight = [1.0, 2.0, 1.0, 1.0]

    input_host = neural_net_text_input(len(init_weight), "input_host")
    input_guest = neural_net_text_input(len(init_weight), "input_guest")
    y = neural_net_label_input(2)

    weight_matrix = neural_net_weight_tensor(init_weight)
    similarity = similarity_matrix(weight_matrix, input_host, input_guest)
    cost = get_probabilities_cost(similarity, y)
    optimizer = get_optimizer_single(cost)

    accuracy = get_accuracy(similarity, y)
    all_params = tf.trainable_variables()
    # variable组成的list

    print('Checking the Training on a Single Batch...')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())

        param_num = sum([np.prod(sess.run(tf.shape(v)))
                         for v in all_params])

        print('There are {} variables in the model'.format(param_num))

        # Training cycle
        for epoch in range(epochs):
            sess.run(optimizer, feed_dict={
                input_host: host,
                input_guest: guest,
                y: labels
            })

            loss = sess.run(cost, feed_dict={
                input_host: host,
                input_guest: guest,
                y: labels
            })
            train_acc = sess.run(accuracy, feed_dict={
                input_host: host,
                input_guest: guest,
                y: labels
            })
            weights = sess.run(weight_matrix)

            if epoch % 100 == 0:
                print('Epoch {:>2}:  '.format(epoch + 1), end='')
                print('Loss: {:>10.4f} Training Accuracy: {:.6f}'.format(
                    loss, train_acc))
                print(weights, end='\n\n')


if __name__ == '__main__':
    train_batch(500)
