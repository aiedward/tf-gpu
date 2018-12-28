import tensorflow as tf


def neural_net_text_input(text_vector_len):
    """
    Return a Tensor for a batch of text input
    : text_vector_len: Length of text vector
    : return: Tensor for text vector input.
    """
    return tf.placeholder(tf.float32, [None, text_vector_len], name='x')


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return tf.placeholder(tf.float32, [None, n_classes], 'y')


def tf_concat_test():
    a = tf.Variable([4, 5, 6])

    b = tf.Variable([1, 2, 3])
    c = tf.concat(values=[a, b], axis=0)

    d = tf.constant([7, 8, 9])
    e = tf.concat(values=[a, d], axis=0)

    f = tf.concat(values=[[a], [d]], axis=0)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        print(sess.run(c))
        print(sess.run(e))
        print(sess.run(f))


def neural_net_weight_tensor_part_test():
    d = 3
    idx = 0
    weight = 100.0
    diag = tf.Variable([weight])
    diag_before = [0.0] * idx
    diag_before = tf.constant(diag_before)
    diag_after = [0.0] * (d - 1 - idx)
    diag_after = tf.constant(diag_after)
    row = tf.concat(values=[diag_before, diag, diag_after], axis=0)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        print(sess.run(row))


def neural_net_weight_tensor(word_weights):
    """
    Return a Tensor for the weight matrix
    :param word_weights: List<Double>
    :return: A matrix tensor
    """
    d = len(word_weights)
    matrix = []

    for idx, weight in enumerate(word_weights):
        diag = tf.Variable([weight])
        diag_before = [0.0]*idx
        diag_before = tf.constant(diag_before)
        diag_after = [0.0]*(d-1-idx)
        diag_after = tf.constant(diag_after)
        row = tf.concat(values=[diag_before, diag, diag_after], axis=0)
        matrix.append(row)

    matrix = [[row] for row in matrix]
    matrix = tf.concat(values=matrix, axis=0)

    return matrix


def neural_net_weight_tensor_test():

    matrix = neural_net_weight_tensor([100.0, 10.0, 1.0])

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        print(sess.run(matrix))


if __name__ == '__main__':

    neural_net_weight_tensor_test()
