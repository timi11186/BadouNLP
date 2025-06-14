import math
import os
from collections import defaultdict
import re

class NewWordDetect:
    def __init__(self, corpus_path):
        # 获取脚本所在目录的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 将相对路径转换为绝对路径
        corpus_path = os.path.join(script_dir, corpus_path) 
        self.max_word_length = 5 # 最大词长
        self.word_count = defaultdict(int) # 词频统计
        self.left_neighbor = defaultdict(dict) # 左邻熵
        self.right_neighbor = defaultdict(dict) # 右邻熵
        self.load_corpus(corpus_path) # 加载语料数据，并进行统计
        self.calc_pmi() # 计算互信息
        self.calc_entropy() # 计算左右熵
        self.calc_word_values() # 计算词值


    #加载语料数据，并进行统计
    def load_corpus(self, path):
        # 打开语料文件，并进行统计
        with open(path, encoding="utf8") as f:
            for line in f:
                sentence = line.strip()
                sentence = re.sub("([^\u4e00-\u9fa5])", '', sentence) # 去除符号，只保留汉字
                for word_length in range(1, self.max_word_length): # 按照窗口长度取词
                    self.ngram_count(sentence, word_length) # 统计词频
        return

    #按照窗口长度取词,并记录左邻右邻
    def ngram_count(self, sentence, word_length):
        # 根据最大词长来进行取词，并记录词频和左右邻字符
        for i in range(len(sentence) - word_length + 1):
            word = sentence[i:i + word_length] # 取词
            self.word_count[word] += 1
            if i - 1 >= 0:
                char = sentence[i - 1] # 左邻字符
                self.left_neighbor[word][char] = self.left_neighbor[word].get(char, 0) + 1
            if i + word_length < len(sentence):
                char = sentence[i +word_length] # 右邻字符
                self.right_neighbor[word][char] = self.right_neighbor[word].get(char, 0) + 1
        return

    #计算熵，word_count_dict 是词频统计字典
    def calc_entropy_by_word_count_dict(self, word_count_dict):
        total = sum(word_count_dict.values()) # 统计词频
        entropy = sum([-(c / total) * math.log((c / total), 10) for c in word_count_dict.values()]) # 计算熵
        return entropy

    #计算左右熵
    def calc_entropy(self):
        self.word_left_entropy = {}
        self.word_right_entropy = {}
        for word, count_dict in self.left_neighbor.items():
            self.word_left_entropy[word] = self.calc_entropy_by_word_count_dict(count_dict)
        for word, count_dict in self.right_neighbor.items():
            self.word_right_entropy[word] = self.calc_entropy_by_word_count_dict(count_dict)


    #统计每种词长下的词总数
    def calc_total_count_by_length(self):
        self.word_count_by_length = defaultdict(int)
        for word, count in self.word_count.items():
            self.word_count_by_length[len(word)] += count
        return

    # 计算互信息（内部凝固度）：PMI = log(P(x,y)/P(x)P(y))是用来衡量词内部字符之间的关联程度的
    # 互信息越大，表示词内部字符之间的关联程度越高，是词都肯定就很高，这就是计算内部凝固度
    def calc_pmi(self):
        self.calc_total_count_by_length() # 统计每种词长下的词总数
        self.pmi = {} # 互信息
        # word, count
        for word, count in self.word_count.items(): # 遍历词频
            p_word = count / self.word_count_by_length[len(word)] # 计算词频
            p_chars = 1
            for char in word:
                p_chars *= self.word_count[char] / self.word_count_by_length[1]
            self.pmi[word] = math.log(p_word / p_chars, 10) / len(word)
        return

    # 计算词值
    def calc_word_values(self):
        self.word_values = {}
        for word in self.pmi: # 遍历词频
            # 如果词长小于2或包含逗号，则跳过
            if len(word) < 2 or "，" in word:
                continue
            pmi = self.pmi.get(word, 1e-3) # 获取词频
            le = self.word_left_entropy.get(word, 1e-3) # 获取左熵
            re = self.word_right_entropy.get(word, 1e-3) # 获取右熵
            self.word_values[word] = pmi * max(le, re) # 词值 = 词频 * (左熵和右熵的最大的那个)
            # self.word_values[word] = pmi * (le + re) # 词值 = 词频 * (左熵和右熵的和)

if __name__ == "__main__":
    # 判断 category_corpus 目录下，有多少个 text 文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(script_dir, "category_corpus") 
    for file in os.listdir(corpus_path):
        nwd = NewWordDetect(os.path.join(corpus_path, file))
        value_sort = sorted([(word, count) for word, count in nwd.word_values.items()], key=lambda x:x[1], reverse=True)
    
    # 将结果写入字词表中，并写入到 category_corpus 目录下
    with open(os.path.join(script_dir, "word_table.txt"), "w", encoding="utf8") as f:
        for word, count in value_sort:
            f.write(f"{word} {count}\n")
    print("写入完成")

    # nwd = NewWordDetect(corpus_path)
    # print(nwd.word_count)
    # print(nwd.left_neighbor)
    # print(nwd.right_neighbor)
    # print(nwd.pmi)
    # print(nwd.word_left_entropy)
    # print(nwd.word_right_entropy)
    # 计算词值,进行分类。将词值从大到小排序
    # value_sort = sorted([(word, count) for word, count in nwd.word_values.items()], key=lambda x:x[1], reverse=True)
    # print(value_sort)
    # print('--------------------------------')
    # print([x for x, c in value_sort if len(x) == 2][:30])
    # print('--------------------------------')
    # print([x for x, c in value_sort if len(x) == 3][:30])
    # print('--------------------------------')
    # print([x for x, c in value_sort if len(x) == 4][:30])

