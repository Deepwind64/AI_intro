import logging
import math
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from sklearn.datasets import fetch_openml
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
        self.acc_log = []
        self.test_feature = test_feature
        self.test_label = test_label

    def train_all(self, features: np.ndarray, labels: np.ndarray, num_iterations=100, eta=0.1):
        def change_weight(example, label):
            for num in range(10):
                z = np.dot(example, self.classifier_weights[num])
                if num == label:
                    if z < 0:
                        self.classifier_weights[num] += eta * example.reshape(self.classifier_weights[num].shape)
                else:
                    if z >= 0:
                        self.classifier_weights[num] -= eta * example.reshape(self.classifier_weights[num].shape)

        for i in tqdm(range(1, num_iterations + 1), desc="train"):
            index = random.randint(0, features.shape[1])
            example = features[index]
            label = labels[index]
            change_weight(example, label)

            if math.log2(i).is_integer():
                self.acc_log.append((i, self.test()))

        # 若训练完最后一次没被测试，则补充测试
        if num_iterations != self.acc_log[-1][0]:
            self.acc_log.append((num_iterations, self.test()))

        logging.info("Training finishes, weights saving.")
        with open("weights.pickle", "wb") as f:
            pickle.dump(self.classifier_weights, f)

    def classify(self, feature) -> int:
        for i in range(10):
            if np.dot(feature, self.classifier_weights[i]) > 0:
                return i
        return -1

    def test(self, feature=None, label=None):
        feature = feature if feature else self.test_feature
        label = label if label else self.test_label
        size = len(feature)
        count = 0
        for i in range(size):
            if self.classify(feature[i]) == label[i]:
                count += 1
        return count / size

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


if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: (%(asctime)s) - %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
    train_x, test_x, train_y, test_y = prepare_dataset()
    classifier = Classifier(784, test_x, test_y)

    logging.info("Training starting...")
    classifier.train_all(train_x, train_y, 10 ** 6, eta=0.05)
    classifier.show_train_acc()
