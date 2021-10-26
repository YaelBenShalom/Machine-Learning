import numpy as np
from generate_regression_data import generate_regression_data
from polynomial_regression import PolynomialRegression
from metrics import mean_squared_error
import matplotlib.pyplot as plt
import random

import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_blobs


def polynomial_regression():
    mse_train = []
    mse_test = []
    N = 100
    max_degree = 10
    random_list_size = 10
    d = 4
    x, y = generate_regression_data(d, N, amount_of_noise=0.1)
    x_train, y_train = np.zeros(
        (random_list_size, 1)), np.zeros((random_list_size, 1))
    x_test, y_test = np.zeros(
        (N - random_list_size, 1)), np.zeros((N - random_list_size, 1))
    random_list = []
    for i in range(0, random_list_size):
        n = random.randint(0, N - 1)
        while n in random_list:
            n = random.randint(0, N - 1)
        random_list.append(n)

    counter_train = 0
    counter_test = 0
    for i in range(N):
        if i in random_list:
            x_train[counter_train] = x[i]
            y_train[counter_train] = y[i]
            counter_train += 1
        else:
            x_test[counter_test] = x[i]
            y_test[counter_test] = y[i]
            counter_test += 1

    for degree in range(max_degree):
        p = PolynomialRegression(degree)
        p.fit(x_train, y_train)
        y_hat_train = p.predict(x_train)
        y_hat_test = p.predict(x_test)
        if len(mse_train) == 0 or min(mse_train) > mean_squared_error(y_train, y_hat_train):
            min_train_y_predict = y_hat_train
        if len(mse_test) == 0 or min(mse_test) > mean_squared_error(y_test, y_hat_test):
            min_test_y_predict = y_hat_test
        mse_train.append(mean_squared_error(y_train, y_hat_train))
        mse_test.append(mean_squared_error(y_test, y_hat_test))

        # p.visualize(x_test, y_test)

    # Q1A
    plt.figure()
    plt.plot(range(max_degree), mse_train,
             color='orange', label='The train error')
    plt.plot(range(max_degree), mse_test, color='blue', label='The test error')
    plt.title('error vs degree')
    plt.xlabel('degree')
    plt.ylabel('error')
    plt.yscale('log')
    plt.legend(loc="best")
    plt.savefig("Q1A.png")

    # Q1B
    features_sorted = np.zeros(x_train.shape)
    targets_sorted = np.zeros(min_train_y_predict.shape)
    sort_indexes = x_train.argsort(axis=0)
    for i in range(len(x_train.argsort(axis=0))):
        features_sorted[i] = x_train[sort_indexes[i]]
        targets_sorted[i] = min_train_y_predict[sort_indexes[i]]

    features2_sorted = np.zeros(x_test.shape)
    targets2_sorted = np.zeros(min_test_y_predict.shape)
    sort_indexes = x_test.argsort(axis=0)
    for i in range(len(x_test.argsort(axis=0))):
        features2_sorted[i] = x_test[sort_indexes[i]]
        targets2_sorted[i] = min_test_y_predict[sort_indexes[i]]

    plt.figure()
    plt.scatter(x_train, y_train, color='blue')
    plt.plot(features_sorted, targets_sorted, color='orange',
             label='The lowest training error')
    plt.plot(features2_sorted, targets2_sorted,
             color='green', label='The lowest testing error')
    plt.title('X vs Y')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
    plt.savefig("Q1B.png")

    # Q5
    # we create 50 separable points
    X, Y = make_blobs(n_samples=50, centers=2,
                      random_state=0, cluster_std=0.60)

    # fit the model
    clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=200)
    clf.fit(X, Y)

    # plot the line, the points, and the nearest vectors to the plane
    xx = np.linspace(-1, 5, 10)
    yy = np.linspace(-1, 5, 10)

    X1, X2 = np.meshgrid(xx, yy)
    Z = np.empty(X1.shape)
    for (i, j), val in np.ndenumerate(X1):
        x1 = val
        x2 = X2[i, j]
        p = clf.decision_function([[x1, x2]])
        Z[i, j] = p[0]
    levels = [-1.0, 0.0, 1.0]
    linestyles = ['dashed', 'solid', 'dashed']
    colors = 'k'
    cs = plt.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired,
                edgecolor='black', s=20, label='data points')
    cs.collections[0].set_label('h(x)=0')
    plt.axis('tight')
    plt.title('Linear Classification Example')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc="best")
    plt.savefig("Q5.png")


if __name__ == "__main__":
    polynomial_regression()
