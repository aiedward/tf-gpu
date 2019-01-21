from collections import defaultdict

import tensorflow as tf
import numpy as np

import config
from vocab_v2 import Vocab
from word_vector_v1 import embed
from word_vector_v1 import text_to_ids
from weight_model_v2 import neural_net_label_input
from similarity_fit_v3 import load_data
from similarity_fit_v3 import get_tokens
from similarity_fit_v3 import pad_sentence_batch


def get_dummy(seed_labels, train_labels):
    dummy_multiple = np.zeros((len(train_labels), len(seed_labels)))
    label_dict = defaultdict(list)
    for idx, label in enumerate(seed_labels):
        label_dict[label] = idx
    for idx, label in enumerate(train_labels):
        dummy_multiple[idx, label_dict[label]] = 1
    return dummy_multiple


def idx2label(label_array, idx):
    keys = list(range(len(label_array)))
    values = label_array
    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1
    )
    out = table.lookup(tf.to_int32(idx))
    return out


def get_argmax(similarity):
    return tf.argmax(similarity, 1)


def get_predict_label(seed_label, the_arg):
    return idx2label(seed_label, the_arg)


def get_accuracy(predict_label, test_label):
    correct_pred = tf.equal(predict_label, tf.constant(test_label))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def train_batch():

    source_path = 'data/sample2.txt'
    source_text = load_data(source_path)
    print("source_text:", source_text)

    initial_words = get_tokens(source_text)
    vocab = Vocab(initial_tokens=initial_words)
    vocab.load_pretrained_embeddings(config.embedding_path_air)

    sentence_ids = text_to_ids(source_text, vocab.token2id)
    sentence_ids = pad_sentence_batch(sentence_ids, vocab.token2id['<blank>'])
    # 常量
    print("sentence_ids:", sentence_ids)

    sentence_place = tf.placeholder(tf.int32, [None, None])
    embed_sentences = embed(vocab, sentence_place)
    # embed_sentences = tf.nn.l2_normalize(embed_sentences, axis=1)

    host = embed_sentences[:3]
    guest = embed_sentences[3:]

    similarity = tf.matmul(guest, tf.transpose(host))
    similarity = tf.identity(similarity, name='similarity')
    probabilities = tf.nn.softmax(similarity)

    labels = [0, 1, 0, 0, 0, 1, 1]
    train_labels_on_seed = get_dummy(labels[:3], labels[3:])
    y = neural_net_label_input(3)

    the_arg_max = get_argmax(similarity)
    pre = get_predict_label(labels[:3], the_arg_max)
    acc = get_accuracy(pre, labels[3:])

    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        train_acc = sess.run(acc, feed_dict={
            sentence_place: sentence_ids,
            y: train_labels_on_seed
        })
        prob = sess.run(probabilities, feed_dict={
            sentence_place: sentence_ids
        })

        print('Training Accuracy: {:.6f}'.format(train_acc))
        print("prob: ", prob, end='\n\n')


if __name__ == "__main__":
    train_batch()
