#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

# TODO: understand that you should not need any other imports other than those already in this file; if you import something that is not installed by default on the csug machines, your code will crash and you will lose points

NUM_FEATURES = 124  # features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = '/u/cs246/data/adult/'  # TODO: if you are working somewhere other than the csug server, change this to the directory where a7a.train, a7a.dev, and a7a.test are on your machine


# %%
# returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature - 1] = value
    x[-1] = 1  # bias
    return y, x


# %%
# return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals], [v[1] for v in vals])
        return np.asarray(ys), np.asarray(
            xs)  # returns a tuple, first is an array of labels, second is an array of feature vectors


# %%
def perceptron(train_ys, train_xs, dev_ys, dev_xs, args):
    weights = np.zeros(NUM_FEATURES)
    # TODO: implement perceptron algorithm here, respecting args

    if args.nodev is True:

        for iter in range(args.iterations):
            for n in range(len(train_xs)):
                if train_ys[n] * (np.dot(weights, train_xs[n])) <= 0:
                    weights += args.lr * train_ys[n] * train_xs[n]

            # predict = np.dot(train_xs, weights)

            # for i in range(len(predict)):
            #     if predict[i] > 0 :
            #         predict[i]= 1
            #     elif predict[i] < 0 :
            #         predict[i]= -1
            #     else:
            #         predict[i]=0

            # if np.all(predict == train_ys):
            #     break

    if args.nodev is False:

        dev_accuracy = []
        accuracy = []
        iteration = []

        for i in range(10):
            for iter in range(args.iterations):
                for n in range(len(train_xs)):
                    if train_ys[n] * (np.dot(weights, train_xs[n])) <= 0:
                        weights += 2 * train_ys[n] * train_xs[n]

            accuracy.append(test_accuracy(weights, train_ys, train_xs))
            dev_accuracy.append(test_accuracy(weights, dev_ys, dev_xs))
            iteration.append(i + 1)

        plt.figure(figsize=(15, 7))
        plt.plot(iteration, accuracy, marker='o', linestyle='-', label="Acuracy of training set")
        plt.plot(iteration, dev_accuracy, marker='o', linestyle='-', label="Acuracy of dev set")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    return weights


# %%
def test_accuracy(weights, test_ys, test_xs):
    accuracy = 0.0
    # TODO: implement accuracy computation of given weight vector on the test data (i.e. how many test data points are classified correctly by the weight vector)

    predict = np.dot(test_xs, weights)

    for i in range(len(predict)):
        if predict[i] > 0:
            predict[i] = 1
        elif predict[i] < 0:
            predict[i] = -1
        else:
            predict[i] = 0

    accuracy = sum(predict == test_ys) / len(test_ys)

    return accuracy


# %%
def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Basic perceptron algorithm.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate to use for update in training loop.')
    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH, 'a7a.train'),
                        help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH, 'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH, 'a7a.test'), help='Test data file.')
    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.train_file: str; file name for training data.
    args.dev_file: str; file name for development data.
    args.test_file: str; file name for test data.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs = parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)
    weights = perceptron(train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(weights, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))
    print('Feature weights (bias last): {}'.format(' '.join(map(str, weights))))


if __name__ == '__main__':
    main()