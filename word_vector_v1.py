"""
@Project   : tf-gpu
@Module    : word_vector_v1.py
@Author    : Jeff [arfu.guo@gmail.com]
@Created   : 2018/12/31 下午10:32
@Desc      : 
"""
import tensorflow as tf
from vocab import Vocab


def text_to_ids(source_text, source_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :return: A list source_id_text
    """
    source_text_ids = [[source_vocab_to_int.get(
        word, source_vocab_to_int['<UNK>']) for word in line.split(' ')]
        for line in source_text.split('\n')]
    return source_text_ids


def embed(p_place):
    """

    :param p_place: placeholder
    p = tf.placeholder(tf.int32, [None, None])
    batch_size and sentence length
    :return: a tensor with variable and placeholder
    """
    vocab = Vocab()
    with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
        # 此处指定使用cpu
        # 第一次建立这个variable_scope, 而不是reuse
        # reuse中最重要的是模型中的trainable variable的复用
        word_embeddings = tf.get_variable(
            'word_embeddings',
            shape=(vocab.size(), vocab.embed_dim),
            initializer=tf.constant_initializer(vocab.embeddings),
            trainable=True
        )

        w_emb = tf.nn.embedding_lookup(word_embeddings, p_place)
        s_emb = tf.reduce_mean(w_emb, axis=1)  # 0 or 1?
        # Difference between np.mean and tf.reduce_mean in Numpy
        # and Tensorflow?
        # https://stackoverflow.com/questions/34236252/difference-between-np-mean-and-tf-reduce-mean-in-numpy-and-tensorflow
        return s_emb
