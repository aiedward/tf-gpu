import tensorflow as tf


def neural_net_text_input(text_vector_len):
    """
    Return a Tensor for a batch of text input
    : text_vector_len: Length of text vector
    : return: Tensor for text vector input.
    """
    return tf.placeholder(tf.float32, [None, text_vector_len], name='x')


def neural_net_text_input_x1(text_vector_len):
    return tf.placeholder(tf.float32, [None, text_vector_len], name='x1')


def neural_net_text_input_x2(text_vector_len):
    return tf.placeholder(tf.float32, [None, text_vector_len], name='x2')


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


def neural_net_weight_tensor_test():

    matrix = neural_net_weight_tensor([100.0, 10.0, 1.0])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(matrix))


init_weight = [100.0, 10.0, 1.0, 1.0]

input_host = neural_net_text_input_x1(len(init_weight))


def vector_transform_x1():
    """
    Convert a matrix of sentence vectors to another matrix
    :return: a matrix tensor
    """
    weight_matrix = neural_net_weight_tensor(init_weight)
    output = tf.matmul(input_host, weight_matrix)

    return output


input_guest = neural_net_text_input_x2(len(init_weight))


def vector_transform_x2():
    """
    Convert a matrix of sentence vectors to another matrix
    :return: a matrix tensor
    """
    weight_matrix = neural_net_weight_tensor(init_weight)
    output = tf.matmul(input_guest, weight_matrix)

    return output


def similarity_matrix():
    """
    给一组句子host，给一组句子guest，算guest和host的相似度
    :return: a tensorflow matrix tensor
    """
    output_host = vector_transform_x1()
    output_guest = vector_transform_x2()
    similarity = tf.matmul(output_guest, tf.transpose(output_host))

    return similarity


n_classes = 2

y = neural_net_label_input(n_classes)


def get_optimizer_single():
    """
    host中每个类别只有一个句子
    :return: optimizer
    """

    similarity = similarity_matrix()

    logits = tf.identity(similarity, name='logits')
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    return optimizer


def train_neural_network(session, optimizer, host_batch, guest_batch,
                         label_batch):

    session.run(optimizer, feed_dict={
        input_host: host_batch,
        input_guest: guest_batch,
        y: label_batch,
        })


def all_test():

    host = [[1, 0, 1, 0], [1, 1, 1, 1]]
    guest = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]]
    labels = [[1, 0], [1, 0], [0, 1], [0, 1]]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_neural_network(
            sess, get_optimizer_single(), host, guest, labels))


if __name__ == '__main__':

    # neural_net_weight_tensor_test()
    all_test()
