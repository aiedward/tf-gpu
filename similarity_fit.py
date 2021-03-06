"""
@Project   : tf-gpu
@Module    : similarity_fit.py
@Author    : Jeff [arfu.guo@gmail.com]
@Created   : 2019/1/5 下午2:38
@Desc      : 
"""
import os

import numpy as np
import tensorflow as tf

import config
from vocab_v2 import Vocab
from word_vector_v1 import embed
from word_vector_v1 import text_to_ids
from weight_model_v2 import neural_net_label_input


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
    return [sentence + [pad_int] * (max_sentence - len(sentence))
            for sentence in sentence_batch]


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

    source_path = 'data/sample.txt'
    source_text = load_data(source_path)
    print("source_text:", source_text)

    vocab = Vocab()
    vocab.load_vocab_from_embedding(config.embedding_path)
    vocab.load_pretrained_embeddings(config.embedding_path)

    sentence_ids = text_to_ids(source_text, vocab.token2id)
    sentence_ids = pad_sentence_batch(sentence_ids, vocab.token2id['<blank>'])
    # 常量
    print("sentence_ids:", sentence_ids)

    sentence_place = tf.placeholder(tf.int32, [None, None])
    embed_sentences = embed(vocab, sentence_place)
    embed_sentences = tf.nn.l2_normalize(embed_sentences, axis=1)

    host = embed_sentences[:2]
    guest = embed_sentences[2:]

    similarity = tf.matmul(guest, tf.transpose(host))
    similarity = tf.identity(similarity, name='similarity')
    probabilities = tf.nn.softmax(similarity)

    labels = [[1, 0], [1, 0], [0, 1], [0, 1]]

    y = neural_net_label_input(2)

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
                sentence_place: sentence_ids,
                y: labels
            })

            loss = sess.run(cost, feed_dict={
                sentence_place: sentence_ids,
                y: labels
            })
            train_acc = sess.run(accuracy, feed_dict={
                sentence_place: sentence_ids,
                y: labels
            })
            prob = sess.run(probabilities, feed_dict={
                sentence_place: sentence_ids
            })

            if epoch % 100 == 0:
                print('Epoch {:>2}:  '.format(epoch + 1), end='')
                print('Loss: {:>10.4f} Training Accuracy: {:.6f}'.format(
                    loss, train_acc))
                print("prob: ", prob, end='\n\n')


if __name__ == '__main__':
    train_batch(500)
