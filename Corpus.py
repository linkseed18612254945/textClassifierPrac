import os
import pyltp
import re
import Bow
import logging
import numpy as np
import math
import nltk
from collections import Counter


class FileItem:
    """ 文档类，每个文档被存储为一个文档类，默认保存有词序列、文档类别、文档id、词性序列四个属性。可以根据需求添加新属性"""

    def __init__(self, id, cate, words, pos):
        self.id = id
        self.cate = cate
        self.words = words
        self.pos = pos

    def __repr__(self):
        return '{0} {1}\n{2}\n{3}\n'.format(self.id, self.cate, self.words, self.pos)


class WordItem:
    """ 词属性类，保存有一个词出现的文档号集合以及一个词在某类里出现的总频次 """

    def __init__(self):
        self.file_ids = Counter()
        self.count = 0
        self.chi = 0

    def add_word(self, file_id):
        self.file_ids.update([file_id])
        self.count += 1

    def __repr__(self):
        return str(self.count) + ' ' + str(self.file_ids)


class Corpus:
    """ 
    语料库对象负责将原始文档读入内存，完成分词和词性标注操作，同时创建对应的词袋模型
    原始语料格式，每个待分类文档保存为一个独立的文件(默认为txt格式，可指定读取特定后缀)。一类文档放在同一个文件夹中，文件夹名即为类别名称。
    通过file_vectors()方法得到向量化后的文档，然后可以进行后续的分类训练
    """

    def __init__(self, path, text_language='ch', file_end='txt'):
        self.category_ids = {}
        self.dir_path = path
        self.file_end = '.' + file_end
        self.file_total_num = len(self.file_paths())
        if text_language == 'ch':
            self.files = self.build_ch_files()
        elif text_language == 'en':
            self.files = self.build_en_files()

    def file_paths(self):
        """ 遍历指定目录，获取文档路径 """
        return [os.path.join(dirname, file) for (dirname, dirs, files) in os.walk(self.dir_path) for file in files
                if file.lower().endswith(self.file_end)]

    def build_ch_files(self):
        """ 遍历原始文档，进行分词词性标注，去除停用词等，创建FileItem类集合 """
        files = []
        category_id = 0
        segmentor = pyltp.Segmentor()
        segmentor.load(r'C:\Users\51694\PycharmProjects\paper\ltp_model\cws.model')
        postagger = pyltp.Postagger()
        postagger.load(r'C:\Users\51694\PycharmProjects\paper\ltp_model\pos.model')
        for ids, path in enumerate(self.file_paths()):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    category = self.path2category(path)
                    if category not in self.category_ids:
                        self.category_ids[category] = category_id
                        category_id += 1
                    raw = self.process_line(f.read())
                    words = self.remove_stop_words(list(segmentor.segment(raw)))
                    words = self.clean_specific(words)
                    pos = list(postagger.postag(words))
                    file = FileItem(ids, category, words, pos)
                    files.append(file)
                except UnicodeDecodeError:
                    logging.warning(path + ' UTF-8解码失败，请检查文本格式')
                    continue
        segmentor.release()
        postagger.release()
        return files

    def build_en_files(self):
        files = []
        category_id = 0
        for ids, path in enumerate(self.file_paths()):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    category = self.path2category(path)
                    if category not in self.category_ids:
                        self.category_ids[category] = category_id
                        category_id += 1
                    raw = self.process_line(f.read(), '')
                    print(raw)
                    words = nltk.word_tokenize(raw)
                    words = [w for w in words if w.isalpha() and w not in nltk.corpus.stopwords.words('english')]
                    print(words)
                    pos = [x[1] for x in nltk.pos_tag(words)]
                    file = FileItem(ids, category, words, pos)
                    files.append(file)
                except UnicodeDecodeError:
                    logging.warning(path + ' UTF-8解码失败，请检查文本格式')
                    continue
        return files

    def build_bow(self):
        """ 利用已创建的FileItem对象集合创建语料库对应的词袋对象 """
        bow_dict = {}
        bow_file_count = Counter()
        for file in self.files:
            bow_file_count[file.cate] += 1
            if file.cate not in bow_dict:
                bow_dict[file.cate] = {}
            for word in file.words:
                if word not in bow_dict[file.cate]:
                    bow_dict[file.cate][word] = WordItem()
                bow_dict[file.cate][word].add_word(file.id)
        return Bow.BagOfWords(bow_dict, bow_file_count)

    def files_data(self, bow, weigh_model='TF-IDF'):
        """ 获取向量化后的文档和对应类别标签数据，可以利用file_num参数指定文档数量，feature_mode指定向量化方法。该方法是提供训练使用的API。 """
        files = self.files
        file_vectors = []
        file_labels = []

        for file in files:
            file_vectors.append(self.__file_to_vector(file, bow, weigh_model))
            file_labels.append(self.category_ids[file.cate])
        return file_vectors, file_labels

    def __word_weight(self, word, file, bow, weigh_model='AllOne'):
        if weigh_model == 'AllOne':
            return 1
        elif weigh_model == 'TF':
            return bow.dict[file.cate][word].file_ids[file.id]
        elif weigh_model == 'TF-IDF':
            tf = bow.dict[file.cate][word].file_ids[file.id] / len(file.words)
            idf = math.log(self.file_total_num / len(bow.dict[file.cate][word].file_ids))
            return tf * idf

    def __file_to_vector(self, file, bow, weigh_model):
        """ 将文档对象向量化 """
        feature_model = 'Frequent_number'
        bow_feature = bow.bow_features(feature_model)
        file_vector = np.zeros(len(bow_feature))
        for word in file.words:
            if word in bow_feature:
                file_vector[bow_feature[word]] = self.__word_weight(word, file, bow, weigh_model)
        return file_vector

    @staticmethod
    def process_line(line, sep=' '):
        """ 去除原始文本中的特殊符号，提高分词准确度 """
        return re.sub("]-·[\s+\.\!\/_,$%^*(+\"\':]+|[+——！，。？、~@#￥%……&*（）():\"=《\\n]+", sep, line)

    @staticmethod
    def remove_stop_words(words):
        """ 针对所有任务都进行的去除停用词和数字 """
        with open('stopwords.txt', 'r', encoding='utf-8') as f:
            stop_words = set(f.read().splitlines())
        return [word for word in words if word not in stop_words and not word.isdigit()]

    @staticmethod
    def clean_specific(words):
        """ 针对特定任务对词进行筛选 """
        def is_english(w):
            return all([ord(c) < 128 for c in w])
        words_copy = words.copy()
        for word in words:
            if is_english(word) or len(word) < 2 or len(word) > 6:
                words_copy.remove(word)
        return words_copy

    @staticmethod
    def path2category(path):
        """ 根据文档路径获得文档所属类别 """
        reverse_path = path[::-1]
        if '\\' in path:
            symb = '\\'
        else:
            symb = '/'
        temp = reverse_path[reverse_path.find(symb) + 1:]
        return temp[:temp.find('\\')][::-1]
