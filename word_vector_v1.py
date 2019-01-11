"""
@Project   : tf-gpu
@Module    : word_vector_v1.py
@Author    : Jeff [arfu.guo@gmail.com]
@Created   : 2018/12/31 下午10:32
@Desc      : 
"""
import tensorflow as tf
from vocab_v2 import Vocab
import config


def text_to_ids(source_text, source_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :return: A list source_id_text
    """
    source_text_ids = [[source_vocab_to_int.get(
        word, source_vocab_to_int['<unk>']) for word in line.split(' ')]
        for line in source_text.split('\n')]
    return source_text_ids


def embed(vocab, p_place):
    """

    :param vocab: Vocab()
    :param p_place: placeholder
    p = tf.placeholder(tf.int32, [None, None])
    batch_size and sentence length
    :return: a tensor with variable and placeholder
    """
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
        # s_emb = tf.reduce_sum(w_emb, axis=1)
        s_emb = tf.reduce_mean(w_emb, axis=1)  # 0 or 1?
        # Difference between np.mean and tf.reduce_mean in Numpy
        # and Tensorflow?
        # https://stackoverflow.com/questions/34236252/difference-between-np-mean-and-tf-reduce-mean-in-numpy-and-tensorflow

        s_emb = tf.nn.l2_normalize(s_emb, axis=1)
        return s_emb


if __name__ == "__main__":
    vocab_main = Vocab()
    vocab_main.load_vocab_from_embedding(config.embedding_path_air)
    vocab_main.load_pretrained_embeddings(config.embedding_path_air)

    sentence = "我 是 谁"
    sentence_ids = text_to_ids(sentence, vocab_main.token2id)
    print("sentence_ids:", sentence_ids)

    sentence_place = tf.placeholder(tf.int32, [None, None])
    embed_main = embed(vocab_main, sentence_place)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        vector = sess.run(embed_main, feed_dict={
            sentence_place: sentence_ids
        })
        # print("list of vectors: ", vector)
        print("type of vector:", type(vector[0]))
        print("vector length:", vector[0].shape)
