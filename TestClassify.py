from sklearn import linear_model
import warnings
import time
import os
import logging
import Corpus
import Bow

# 设定基本环境参数
CORPUS_ROOT = r'.\data'
date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    filename='./log/%s.txt' % date_time,
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')



# 设定训练参数
CPU_CORE = 1
KERNEL = 'rbf'
DECISION_FUNCTION_SHAPE = 'ovo'
TRAIN_NUM = 'all'
TEST_NUM = 'all'
MODEL = 'Frequency'
FREQUENCY = 0.5


def correct_accuracy(predict, correct):
    correct_count = 0
    test_num = len(predict)
    for i in range(test_num):
        if predict[i] == correct[i]:
            correct_count += 1
    return correct_count / test_num

if __name__ == '__main__':
    # 读取训练和测试语料库，构建或加载词袋对象
    print('-----------------------------------------------开始数据预处理模块---------------------------------------------')
    logging.info('开始数据预处理模块, 模式为%s, 词频选择为%f' % (MODEL, FREQUENCY))
    start_time = time.clock()
    test_corpus = Corpus.Corpus(os.path.join(CORPUS_ROOT, 'test'))
    print('已读入测试语料......')
    train_corpus = Corpus.Corpus(os.path.join(CORPUS_ROOT, 'train'))
    print('已读入训练语料......')
    bow = train_corpus.build_bow()
    # bow.save_bow()
    print('已加载词袋模型......')
    # 得到向量化的语料和对应类别用于训练分类器
    if TRAIN_NUM == 'all':
        TRAIN_NUM = len(train_corpus.files)
    if TEST_NUM == 'all':
        TEST_NUM = len(test_corpus.files)
    train_files, train_labels = train_corpus.files_data(bow, TRAIN_NUM, MODEL, FREQUENCY)
    test_files, test_labels = test_corpus.files_data(bow, TEST_NUM, MODEL, FREQUENCY)
    print('已向量化%d个训练文档, %d个测试文档, 共%d类, 向量维度为%d' % (TRAIN_NUM, TEST_NUM,
                                                   len(train_corpus.category_ids.keys()), len(test_files[0])))
    logging.info('已向量化%d个训练文档, %d个测试文档, 共%d类, 向量维度为%d' % (TRAIN_NUM, TEST_NUM,
                                                   len(train_corpus.category_ids.keys()), len(test_files[0])))
    end_time = time.clock()
    data_loading_time = end_time - start_time
    print('数据预处理结束，共耗时 %.2f s' % data_loading_time)
    logging.info('数据预处理结束，共耗时 %.2f s' % data_loading_time)

    # 使用sci-kit包训练Logistic模型
    print('----------------------------------------------开始分类器训练模块----------------------------------------------')
    start_time = time.clock()
    logreg = linear_model.LogisticRegression()
    logreg.fit(train_files, train_labels)
    end_time = time.clock()
    learn_time = end_time - start_time
    print('Logistic分类器训练结束，共耗时 %.2f s' % learn_time)
    logging.info('Logistic分类器训练结束，共耗时 %.2f s' % learn_time)

    # 计算每个测试数据的预测值并评价准确率
    print('----------------------------------------------开始测试分类结果------------------------------------------------')
    start_time = time.clock()
    test_predict = logreg.predict(test_files)
    accuracy = correct_accuracy(test_predict, test_labels)
    end_time = time.clock()
    evaluate_time = end_time - start_time
    print('文本分类任务结束, 共测试%d个测试数据, 耗时%.2f s, 正确率为%.3f'
          % (len(test_predict), float(data_loading_time + learn_time + evaluate_time), accuracy * 100) + '%')
    logging.info('文本分类任务结束, 共测试%d个测试数据, 耗时%.2f s, 正确率为%.3f'
          % (len(test_predict), float(data_loading_time + learn_time + evaluate_time), accuracy * 100) + '%')