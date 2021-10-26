import numpy as np
import matplotlib.pyplot as plt
from your_code import load_data, ZeroOneLoss

print('Question 2a')
train_features, test_features, train_targets, test_targets = load_data(
    'synthetic', fraction=1)
ones_mat = np.ones((train_features.shape[0], 1))
train_features = np.append(train_features, ones_mat, axis=1)
loss = ZeroOneLoss()

bias = np.array([0.5, -0.5, -1.5, -2.5, -3.5, -4.5, -5.5])
w = np.array([1, 0])
loss_landscape = np.zeros(len(bias))
i = 0
for b in bias:
    w[1] = b
    loss_landscape[i] = loss.forward(train_features, w, train_targets)
    i += 1

plt.figure()
plt.plot(bias, loss_landscape, color='orange', label='Loss Landscape')
plt.title('Loss Landscape')
plt.xlabel('Bias')
plt.ylabel('Loss')
plt.legend(loc="best")
plt.savefig("Q2a.png")


print('Question 2a')

train_features, test_features, train_targets, test_targets = load_data(
    'synthetic', fraction=1)
train_features_optimized = train_features[[0, 1, 4, 5]]
train_targets_optimized = train_targets[[0, 1, 4, 5]]
ones_mat = np.ones((train_features_optimized.shape[0], 1))
train_features_optimized = np.append(
    train_features_optimized, ones_mat, axis=1)
loss = ZeroOneLoss()

bias = np.array([0.5, -0.5, -1.5, -2.5, -3.5, -4.5, -5.5])
w = np.array([1, 0])
loss_landscape = np.zeros(len(bias))
i = 0
for b in bias:
    w[1] = b
    loss_landscape[i] = loss.forward(
        train_features_optimized, w, train_targets_optimized)
    i += 1

plt.figure()
plt.plot(bias, loss_landscape, color='blue', label='Loss Landscape')
plt.title('Landscape on a set of 4 points')
plt.xlabel('Bias')
plt.ylabel('Loss')
plt.legend(loc="best")
plt.savefig("Q2b.png")
