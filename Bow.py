from collections import OrderedDict
import pickle
import os


class BagOfWords:
    """ 词袋模型类, 每个语料库对应生成一个词袋，词袋类负责支持特征筛选和对词进行编号，对文档类进行向量化 """

    def __init__(self, words_dict=None, file_count=None):
        self.dict = words_dict
        self.file_count = file_count

    def bow_features(self, feature_model, frequency):
        """ 
            特征选择，根据需求选择特定词典作为文档特征, 参数feature_model用于设定特征选择方法
            Total: 将全部词作为训练特征
            Frequency: 选择每个类别中出现频率前 n% 的词作为训练特征，用frequency参数进行设定，默认为50%
        """
        words = set()
        if feature_model == 'Total':
            words = self.__total_words()
        elif feature_model == 'Frequency':
            words = self.__frequency_words(frequency)
        else:
            pass
        return self.dict_with_id(words)

    def __total_words(self):
        chosen_words = []
        for category in self.dict:
            chosen_words += list(self.dict[category].keys())
        return set(chosen_words)

    def __frequency_words(self, frequency):
        chosen_words = []
        for category in self.dict:
            category_num = int(len(self.dict[category].keys()) * frequency)
            chosen_items = sorted(self.dict[category].items(), key=lambda w: w[1].count, reverse=True)[:category_num]
            chosen_words += [item[0] for item in chosen_items]
        return set(chosen_words)

    def __chi_words(self, frequency):
        chosen_words = []
        for category in self.dict:
            category_num = int(len(self.dict[category].keys()) * frequency)
            chosen_items = sorted(self.dict[category].items(), key=lambda w: w[1].count, reverse=True)[:category_num]
            chosen_words += [item[0] for item in chosen_items]

    @staticmethod
    def dict_with_id(words):
        """ 对词进行编号，使一个维度对应一个词 """
        id_dict = OrderedDict()
        for idx, word in enumerate(words):
            id_dict[word] = idx
        return id_dict

    def save_bow(self, bow_name='saved_BoW'):
        """ 存储词袋模型对象在一个新建目录下，包含两个字典数据 """
        os.mkdir(bow_name)
        with open('%s/words_dict' % bow_name, 'wb') as f:
            pickle.dump(self.dict, f)
        with open('%s/file_count' % bow_name, 'wb') as f:
            pickle.dump(self.file_count, f)

    def load_bow(self, bow_name='saved_BoW'):
        """ 读取已存储的词袋模型字典，要求读取目录包含words_dict, file_count两个Pickle字典 """
        try:
            with open('%s/words_dict' % bow_name, 'rb') as f:
                self.dict = pickle.load(f)
            with open('%s/file_count' % bow_name, 'rb') as f:
                self.file_count = pickle.load(f)
        except IOError as e:
            raise e
