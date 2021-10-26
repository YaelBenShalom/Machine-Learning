import numpy


class PriorProbability():
    def __init__(self):
        """
        This is a simple classifier that only uses prior probability to classify 
        points. It just looks at the classes for each data point and always predicts
        the most common class.
        """

        self.most_common_class = None

    def fit(self, features, targets):
        """
        Implement a classifier that works by prior probability. Takes in features
        and targets and fits the features to the targets using prior probability.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N 
                examples.
        Output:
            VOID: You should be updating self.most_common_class with the most common class
            found from the prior probability.
        """
        # Assumes 0/1 array and uses builtin functions
        ones_counter = targets.sum()
        if ones_counter > len(targets)/2:
            # 1 is most frequent
            self.most_common_class = 1
        else:
            # 0 is most frequent
            self.most_common_class = 0

    def predict(self, data):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        Outputs:
            predictions (np.array): numpy array of size N array which has the predicitons 
            for the input data.
        """

        N = data.shape[0]
        # All our predictions are the most common class without any data usage
        predictions = numpy.ones((N, 1))*self.most_common_class

        return predictions
