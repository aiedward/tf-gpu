import numpy as np


class Vocab:
    """
    Implements a vocabulary to store the tokens in the data,
    with their corresponding embeddings.
    """
    def __init__(self, filename=None, initial_tokens=None, lower=False):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}
        self.lower = lower

        self.embed_dim = None
        self.embeddings = None

        self.pad_token = '<blank>'
        self.unk_token = '<unk>'

        self.initial_tokens = initial_tokens if initial_tokens is not None \
            else []
        self.initial_tokens.extend([self.pad_token, self.unk_token])
        for token in self.initial_tokens:
            self.add(token)

        if filename is not None:
            self.load_from_file(filename)

    def size(self):
        """
        get the size of vocabulary
        Returns:
            an integer indicating the size
        """
        return len(self.id2token)

    def load_from_file(self, file_path):
        """
        loads the vocab from file_path
        Args:
            file_path: a file with a word in each line
        """
        for line in open(file_path, 'r'):
            token = line.rstrip('\n')
            self.add(token)

    def load_vocab_from_embedding(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as fin:
            line = next(fin)
            contents = line.strip().split()
            word_num = int(contents[0])
            if self.embed_dim is None:
                self.embed_dim = int(contents[1])
            token_num = 0
            for line in fin:
                contents = line.strip().split()
                if len(contents) != self.embed_dim+1:
                    continue
                token = contents[0]
                token_num += 1
                if token not in self.token2id:
                    self.add(token)

            if token_num != word_num:
                print("word_num in the first row:", word_num)
                print("Number of tokens put in the vocab:", token_num)
                print("Vocab size:", self.size())

    def add(self, token, cnt=1):
        """
        adds the token to vocab
        Args:
            token: a string
            cnt: a num indicating the count of the token to add, default is 1
        """
        token = token.lower() if self.lower else token
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx
        if cnt > 0:
            if token in self.token_cnt:
                self.token_cnt[token] += cnt
            else:
                self.token_cnt[token] = cnt
        return idx

    def load_pretrained_embeddings(self, embedding_path):
        """
        loads the pretrained embeddings from embedding_path,
        tokens not in pretrained embeddings will be filtered
        Args:
            embedding_path: the path of the pretrained embedding file
        """
        trained_embeddings = {}
        with open(embedding_path, 'r', encoding='utf-8') as fin:
            next(fin)
            for line in fin:
                contents = line.strip().split()
                if len(contents) != self.embed_dim+1:
                    continue
                token = contents[0]
                if token not in self.token2id:
                    continue
                trained_embeddings[token] = list(map(float, contents[1:]))
                if self.embed_dim is None:
                    self.embed_dim = len(contents) - 1
        filtered_tokens = trained_embeddings.keys()
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)
        # load embeddings
        self.embeddings = np.zeros([self.size(), self.embed_dim])
        for token in self.token2id.keys():
            if token in trained_embeddings:
                self.embeddings[self.get_id(token)] = trained_embeddings[token]
        print("The pretrained embeddings from {} are loaded".format(
            embedding_path))

    def get_id(self, token):
        """
        gets the id of a token, returns the id of unk token if token is not
        in vocab
        Args:
            token: a string indicating the word
        Returns:
            an integer
        """
        token = token.lower() if self.lower else token
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]


if __name__ == "__main__":
    import config
    vocab_main = Vocab()
    vocab_main.load_vocab_from_embedding(config.embedding_path_air)
    vocab_main.load_pretrained_embeddings(config.embedding_path_air)
