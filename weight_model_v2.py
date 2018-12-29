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


