import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def result(test_files, test_labels, model, train_num):
    predict_labels = [model.predict(test_file) for test_file in test_files]
    evaluate(predict_labels, test_labels, train_num, test_files)


def evaluate(predict_labels, test_labels, train_num, test_files):
    wrong_count = 0
    data_num = len(predict_labels)
    for i in range(data_num):
        if predict_labels[i] != test_labels[i]:
            wrong_count += 1
    rate = 1 - (wrong_count / data_num)
    percent = str(rate * 100)
    print('----------------------------------')
    print('Training data number:', train_num)
    print('Testing data number:', data_num)
    print('correct rate:', percent + '%')


def show_image(image):
    pic = image.reshape(28, 28)
    plt.imshow(pic, cmap='gray')
    plt.show()
