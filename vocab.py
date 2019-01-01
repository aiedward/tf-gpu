"""
@Project   : tf-gpu
@Module    : vocab.py
@Author    : Jeff [arfu.guo@gmail.com]
@Created   : 2019/1/1 下午8:05
@Desc      : 
"""
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
        # count
        self.lower = lower
        # 是否都要转成小写字母，对英文有效

        self.embed_dim = None
        self.embeddings = None

        self.pad_token = '<blank>'
        # padding，把格式补整齐
        self.unk_token = '<unk>'
        # unknown

        # In fasttext word vector, </s> means space?

        self.initial_tokens = initial_tokens if initial_tokens is not None \
            else []
        self.initial_tokens.extend([self.pad_token, self.unk_token])
        for token in self.initial_tokens:
            self.add(token)
            # initial_tokens可以认为是自定义的token

        if filename is not None:
            self.load_from_file(filename)
            # load_from_file是个好名字

    def size(self):
        """
        get the size of vocabulary
        Returns:
            an integer indicating the size
        """
        return len(self.id2token)
        # we check the dict of id2token

    def load_from_file(self, file_path):
        """
        loads the vocab from file_path
        Args:
            file_path: a file with a word in each line
        """
        for line in open(file_path, 'r'):
            token = line.rstrip('\n')
            self.add(token)
            # token从配置文件中获得

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

    def get_token(self, idx):
        """
        gets the token corresponding to idx, returns unk token if idx is
        not in vocab
        Args:
            idx: an integer
        returns:
            a token string
        """
        try:
            return self.id2token[idx]
        except KeyError:
            return self.unk_token

    def add(self, token, cnt=1):
        """
        adds the token to vocab
        Args:
            token: a string
            cnt: a num indicating the count of the token to add, default is 1
        """
        token = token.lower() if self.lower else token
        # 是否都转成小写字母
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.id2token)
            # idx的赋值, index
            self.id2token[idx] = token
            self.token2id[token] = idx
        if cnt > 0:
            if token in self.token_cnt:
                self.token_cnt[token] += cnt
                # 不仅保存token，也保存出现的频数
            else:
                self.token_cnt[token] = cnt
        return idx
        # 返回token对应的index
        # 如果改变了传进来的参数，或者改变了实例属性，返回None是更好的选择？

    def filter_tokens_by_cnt(self, min_cnt):
        """
        filter the tokens in vocab by their count
        Args:
            min_cnt: tokens with frequency less than min_cnt is filtered
        """
        filtered_tokens = [token for token in self.token2id
                           if self.token_cnt[token] >= min_cnt]
        # rebuild the token x id map，重新构建两个字典
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
            # 首先要把padding和unknown加进去
            # cnt=0, 目的是self.token_cnt不做改变
        for token in filtered_tokens:
            self.add(token, cnt=0)

    def randomly_init_embeddings(self, embed_dim):
        """
        randomly initializes the embeddings for each token
        Args:
            embed_dim: the size of the embedding for each token
        """
        self.embed_dim = embed_dim
        self.embeddings = np.random.rand(self.size(), embed_dim)
        # 使用numpy来随机化矩阵
        for token in [self.pad_token, self.unk_token]:
            self.embeddings[self.get_id(token)] = np.zeros([self.embed_dim])
            # 对于padding和unknown，不能随机化，要处理成0

    def load_pretrained_embeddings(self, embedding_path):
        """
        loads the pretrained embeddings from embedding_path,
        tokens not in pretrained embeddings will be filtered
        Args:
            embedding_path: the path of the pretrained embedding file
        """
        trained_embeddings = {}
        with open(embedding_path, 'r') as fin:
            line = next(fin)
            contents = line.strip().split()
            word_num = int(contents[0])
            word_dim = int(contents[1])
            # 对于fasttext word vector，理论上会用掉700M内存
            for line in fin:
                contents = line.strip().split()
                token = contents[0]
                if token not in self.token2id:
                    continue
                trained_embeddings[token] = list(map(float, contents[1:]))
                # In fasttext word vector, the format applies
                # python 3中的float是float64，相当于c语言中的双精度浮点型
                # 谈谈关于Python里面小数点精度控制的问题
                # https://www.cnblogs.com/herbert/p/3402245.html
                # Numpy float64 vs Python float
                # https://stackoverflow.com/questions/27098529/numpy-float64-vs-python-float
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
        self.embeddings = np.zeros([word_num, word_dim])
        # self.embeddings = np.zeros([self.size(), self.embed_dim])
        for token in self.token2id.keys():
            if token in trained_embeddings:
                self.embeddings[self.get_id(token)] = trained_embeddings[token]
        print("The pretrained embeddings from {} are loaded".format(
            embedding_path))

    def convert_to_ids(self, tokens):
        """
        Convert a list of tokens to ids, use unk_token if the token is not in vocab.
        Args:
            tokens: a list of token
        Returns:
            a list of ids
        """
        vec = [self.get_id(label) for label in tokens]
        return vec

    def recover_from_ids(self, ids, stop_id=None):
        """
        Convert a list of ids to tokens, stop converting if the stop_id is encountered
        Args:
            ids: a list of ids to convert
            stop_id: the stop id, default is None
        Returns:
            a list of tokens
        """
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens


if __name__ == "__main__":
    import config
    Vocab().load_pretrained_embeddings(config.embedding_path)
