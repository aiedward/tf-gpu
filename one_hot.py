from fasttext_similarity_v2 import load_data
from vocab_v2 import Vocab
import config
from word_vector_v1 import text_to_ids
from similarity_fit_v3 import get_tokens


def sentence2embed(vocab, sentence_with_ids):
    embed = [0]*vocab.size()
    for digit in sentence_with_ids:
        embed[digit] = 1
        # if digit not in [vocab.token2id['<unk>'], vocab.token2id['<blank>']]:
        #     embed[digit] = 1
    return embed


def generate_one_hot():

    source_path = 'data/sample.txt'
    source_text = load_data(source_path)
    print("source_text:", source_text)

    initial_words = get_tokens(source_text)
    vocab = Vocab(initial_tokens=initial_words)
    vocab.load_pretrained_embeddings(config.embedding_path)

    sentence_ids = text_to_ids(source_text, vocab.token2id)

    embeddings = [sentence2embed(vocab, sentence) for sentence in sentence_ids]

    print('embeddings: ', embeddings)


if __name__ == "__main__":
    generate_one_hot()
