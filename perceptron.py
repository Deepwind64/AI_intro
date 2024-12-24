import logging
import math
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm


def prepare_dataset():
    logging.info("Dataset downloading...")
    MNIST = fetch_openml('mnist_784', version=1, cache=True, parser='auto')

    features = MNIST.data.to_numpy()
    features = features.astype(np.float32) / 256.0
    labels = MNIST.target.astype(np.int8).to_numpy()
    train_x, test_x = np.split(features, [60000])  # 60000:10000
    train_y, test_y = np.split(labels, [60000])
    return train_x, test_x, train_y, test_y


class Classifier:
    def __init__(self, feature_size, test_feature, test_label):
        self.classifier_weights = np.zeros((10, feature_size), dtype='float64')  # 10组权重
        self.classifier_biases = np.zeros(10, dtype='float64')  # 10个偏置
        self.acc_log = []
        self.test_feature = test_feature
        self.test_label = test_label

    def train_all(self, features: np.ndarray, labels: np.ndarray, num_iterations=100, eta=0.1):
        def train(example, label):
            for num in range(10):
                z = np.dot(example, self.classifier_weights[num]) + self.classifier_biases[num]
                if num == label:
                    if z < 0:
                        self.classifier_weights[num] += eta * example
                        self.classifier_biases[num] += eta
                else:
                    if z >= 0:
                        self.classifier_weights[num] -= eta * example
                        self.classifier_biases[num] -= eta

        for i in tqdm(range(1, num_iterations + 1), desc="train"):
            index = random.randint(0, features.shape[0] - 1)  # 修正索引范围
            example = features[index]
            label = labels[index]
            train(example, label)

            if math.log2(i).is_integer():
                self.acc_log.append((i, self.test()))

        # 若训练完最后一次没被测试，则补充测试
        if num_iterations != self.acc_log[-1][0]:
            self.acc_log.append((num_iterations, self.test()))

        logging.info("Training finishes, weights and biases saving.")
        with open("weights_and_biases.pickle", "wb") as f:
            pickle.dump({'weights': self.classifier_weights, 'biases': self.classifier_biases}, f)

    def classify(self, feature) -> int:
        scores = np.dot(feature, self.classifier_weights.T) + self.classifier_biases
        return np.argmax(scores)

    def test(self, feature=None, label=None):
        feature = feature if feature else self.test_feature
        label = label if label else self.test_label
        size = len(feature)
        count_list = [1 for i in range(size) if self.classify(feature[i]) == label[i]]
        return sum(count_list) / size

    def show_train_acc(self, show=False):
        epochs, accuracies = zip(*self.acc_log)
        plt.figure(figsize=(10, 6))
        plt.xscale('log', base=2)
        plt.plot(epochs, accuracies, marker='o')
        plt.title('Model Acc vs Training Epochs')
        plt.xlabel('Training Epochs (log2)')
        plt.ylabel('Acc')
        texts = [plt.text(epochs[i], acc, f'{acc:.3f}', ha='right', va='bottom') for i, acc in enumerate(accuracies)]
        adjust_text(texts, )
        plt.grid(True, which="both", ls="--")
        if show:
            plt.show()
        plt.savefig("train_acc.png")
        logging.info("Training accuracy img has been saved successfully.")

    def evaluate_model(self):
        predictions = np.array([self.classify(test_x) for test_x in self.test_feature])
        accuracy = accuracy_score(self.test_label, predictions)
        precision = precision_score(self.test_label, predictions, average='macro')
        recall = recall_score(self.test_label, predictions, average='macro')
        f1 = f1_score(self.test_label, predictions, average='macro')

        logging.info(f"Model performance: \nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")


if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: (%(asctime)s) - %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
    train_x, test_x, train_y, test_y = prepare_dataset()
    classifier = Classifier(784, test_x, test_y)

    logging.info("Training starting...")
    classifier.train_all(train_x, train_y, 10 ** 6, eta=0.01)
    classifier.show_train_acc()
    classifier.evaluate_model()
