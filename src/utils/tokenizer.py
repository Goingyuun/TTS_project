# 文件路径: src/utils/tokenizer.py

import re
import json


class SimpleTokenizer:
    def __init__(self, vocab=None, lower=True):
        """
        初始化 Tokenizer
        :param vocab: 可选预先定义的词汇表（dict: token -> id）
        :param lower: 是否将文本转换为小写
        """
        self.lower = lower
        # 定义一些特殊的 token
        self.special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
        self.vocab = {} if vocab is None else vocab
        self.inv_vocab = {}
        if vocab is not None:
            self.build_inv_vocab()

    def build_inv_vocab(self):
        """构建 id 到 token 的反向映射"""
        self.inv_vocab = {i: token for token, i in self.vocab.items()}

    def build_vocab(self, texts, min_freq=1):
        """
        根据给定的文本列表构建词汇表
        :param texts: 文本列表，每个元素为字符串
        :param min_freq: token 出现的最小频率（低于该频率的 token 将被忽略）
        """
        token_freq = {}
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1

        # 初始化词汇表，首先加入特殊 token
        self.vocab = {}
        for token in self.special_tokens:
            self.vocab[token] = len(self.vocab)

        # 添加频率足够的 token
        for token, freq in token_freq.items():
            if freq >= min_freq and token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        self.build_inv_vocab()

    def tokenize(self, text):
        """
        分词函数：对输入文本进行简单的分词（保留标点符号）
        :param text: 输入字符串
        :return: token 列表
        """
        if self.lower:
            text = text.lower()
        # 使用正则表达式实现简单分词：匹配单词或非空白字符
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens

    def text_to_ids(self, text):
        """
        将文本转换为数字 ID 序列
        :param text: 输入字符串
        :return: 数字 ID 列表
        """
        tokens = self.tokenize(text)
        # 可选：在序列前后加上 <sos> 和 <eos> token
        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        return token_ids

    def ids_to_text(self, ids):
        """
        将数字 ID 序列转换回文本
        :param ids: 数字 ID 列表
        :return: 转换后的文本字符串
        """
        tokens = [self.inv_vocab.get(i, "<unk>") for i in ids]
        return " ".join(tokens)

    def save_vocab(self, file_path):
        """保存词汇表到 JSON 文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)

    def load_vocab(self, file_path):
        """从 JSON 文件加载词汇表"""
        with open(file_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.build_inv_vocab()


if __name__ == "__main__":
    # 测试示例
    texts = ["Hello world!", "This is a test.", "Another test, for tokenizer."]
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts, min_freq=1)

    for text in texts:
        print("原始文本:", text)
        tokens = tokenizer.tokenize(text)
        print("分词结果:", tokens)
        ids = tokenizer.text_to_ids(text)
        print("编码后的 ID 序列:", ids)
        recovered_text = tokenizer.ids_to_text(ids)
        print("反向转换结果:", recovered_text)
        print("-" * 40)
