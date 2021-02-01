import numpy as np
import random
import matplotlib.pyplot as plt
from your_code import HingeLoss, SquaredLoss
from your_code import metrics
from your_code import GradientDescent, load_data
from your_code import L1Regularization, L2Regularization

class MultiClassGradientDescentQ3:
    """
    Implements linear gradient descent for multiclass classification. Uses
    One-vs-All (OVA) classification for aggregating binary classification
    results to the multiclass setting.

    Arguments:
        loss - (string) The loss function to use. One of 'hinge' or 'squared'.
        regularization - (string or None) The type of regularization to use.
            One of 'l1', 'l2', or None. See regularization.py for more details.
        learning_rate - (float) The size of each gradient descent update step.
        reg_param - (float) The hyperparameter that controls the amount of
            regularization to perform. Must be non-negative.
    """

    def __init__(self, loss, regularization=None,
                 learning_rate=0.01, reg_param=0.05):
        self.loss = loss
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.reg_param = reg_param

        self.model = []
        self.classes = None

    def fit(self, features, targets, batch_size=None, max_iter=1000):
        """
        Fits a multiclass gradient descent learner to the features and targets
        by using One-vs-All classification. In other words, for each of the c
        output classes, train a GradientDescent classifier to determine whether
        each example does or does not belong to that class.

        Store your c GradientDescent classifiers in the list self.model. Index
        c of self.model should correspond to the binary classifier trained to
        predict whether examples do or do not belong to class c.

        Arguments:
            features - (np.array) An Nxd array of features, where N is the
                number of examples and d is the number of features.
            targets - (np.array) A 1D array of targets of size N. Contains c
                unique values (the possible class labels).
            batch_size - (int or None) The number of examples used in each
                iteration. If None, use all of the examples in each update.
            max_iter - (int) The maximum number of updates to perform.
        Modifies:
            self.model - (list) A list of c GradientDescent objects. The models
                trained to perform OVA classification for each class.
            self.classes - (np.array) A numpy array of the unique target
                values. Required to associate a model index with a target value
                in predict.
        """
        classes_list = []
        for x in targets:
            if x not in classes_list:
                classes_list.append(x)
        self.classes = np.array(classes_list)


        for c in self.classes:
            targets_c = -1 * (np.ones(targets.shape))
            for i in range(len(targets)):
                if targets[i] == c:
                    targets_c[i] = 1

            gradient_descent = GradientDescent(self.loss, self.regularization,
                                                self.learning_rate, self.reg_param)
            gradient_descent.fit(features, targets_c, batch_size, max_iter)
            self.model.append(gradient_descent)
        print("self.classes = ", self.classes)


    def predict(self, features):
        """
        Predicts the class labels of each example in features using OVA
        aggregation. In other words, predict as the output class the class that
        receives the highest confidence score from your c GradientDescent
        classifiers. Predictions should be in the form of integers that
        correspond to the index of the predicted class.

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            predictions - (np.array) A 1D array of predictions of length N,
                where index d corresponds to the prediction of row N of
                features.
        """
        confidence_matrix = np.zeros((features.shape[0], self.classes.shape[0]))
        for i in range(len(self.classes)):
            gradient_descent = self.model[i]
            confidence = gradient_descent.confidence(features)
            confidence_matrix[:, i] = confidence

        predictions = np.zeros((features.shape[0]))
        for r in range(len(confidence_matrix)):
            max_index = np.argmax(confidence_matrix[r], axis=0)
            predictions[r] = self.classes[max_index]

        return predictions


print('Question 3a')
max_iter = 1000
batch_size = 1
fraction = 0.75
learning_rate = 1e-4
reg_param = 0.05
loss = 'squared'
regularization = 'l1'

train_features, test_features, train_targets, test_targets = \
    load_data('mnist-multiclass', fraction=fraction)
learner = MultiClassGradientDescentQ3(loss=loss, regularization=regularization)
learner.fit(train_features, train_targets)
predictions = learner.predict(test_features)

confusion_matrix = metrics.confusion_matrix(test_targets, predictions)
print("confusion_matrix = ", confusion_matrix)