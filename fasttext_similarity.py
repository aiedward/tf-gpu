import os
from vocab_v2 import Vocab
import config
from word_vector_v1 import text_to_ids
from word_vector_v1 import embed
import tensorflow as tf


def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        return f.read()


def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has
    the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def similarity_matrix():

    source_path = 'data/sample.txt'
    source_text = load_data(source_path)
    print("source_text:", source_text)

    vocab = Vocab()
    vocab.load_vocab_from_embedding(config.embedding_path_air)
    vocab.load_pretrained_embeddings(config.embedding_path_air)

    sentence_ids = text_to_ids(source_text, vocab.token2id)
    sentence_ids = pad_sentence_batch(sentence_ids, vocab.token2id['<blank>'])
    # 常量
    print("sentence_ids:", sentence_ids)

    sentence_place = tf.placeholder(tf.int32, [None, None])
    embed_sentences = embed(vocab, sentence_place)
    embed_sentences = tf.nn.l2_normalize(embed_sentences, axis=1)

    # enc_embed_input = tf.contrib.layers.embed_sequence(
    #     sentence_place, source_vocab_size, vocab.embed_dim)
    # tf.contrib.layers.embed_sequence() is for what?
    # https://stackoverflow.com/questions/40784656/tf-contrib-layers-embed-sequence-is-for-what

    host = embed_sentences[:2]
    guest = embed_sentences[2:]
    # labels = [[1, 0], [1, 0], [0, 1], [0, 1]]

    # difference = host[1]-guest[2]

    similarity = tf.matmul(guest, tf.transpose(host))
    similarity = tf.identity(similarity, name='similarity')
    probabilities = tf.nn.softmax(similarity)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        vector = sess.run(embed_sentences, feed_dict={
            sentence_place: sentence_ids
        })
        # print("list of vectors: ", vector)
        print("type of vector:", type(vector[0]))
        print("vector length:", vector[0].shape)

        sim = sess.run(similarity, feed_dict={
            sentence_place: sentence_ids
        })
        print("sim: ", sim)

        prob = sess.run(probabilities, feed_dict={
            sentence_place: sentence_ids
        })
        print("prob: ", prob)

        # diff = sess.run(difference, feed_dict={
        #     sentence_place: sentence_ids
        # })
        # print("diff: ", diff)


if __name__ == "__main__":
    similarity_matrix()
