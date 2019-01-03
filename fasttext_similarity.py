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


def similarity_matrix():

    source_path = 'data/sample.txt'
    source_text = load_data(source_path)
    print("source_text:", source_text)

    vocab = Vocab()
    vocab.load_vocab_from_embedding(config.embedding_path_air)
    vocab.load_pretrained_embeddings(config.embedding_path_air)

    sentence_ids = text_to_ids(source_text, vocab.token2id)
    print("sentence_ids:", sentence_ids)
    sentence_place = tf.placeholder(tf.int32, [None, None])
    embed_sentences = embed(vocab, sentence_place)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        vector = sess.run(embed_sentences, feed_dict={
            sentence_place: sentence_ids
        })
        # print("list of vectors: ", vector)
        print("type of vector:", type(vector[0]))
        print("vector length:", vector[0].shape)


if __name__ == "__main__":
    similarity_matrix()
