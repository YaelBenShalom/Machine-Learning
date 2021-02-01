import numpy as np 
from .distances import euclidean_distances, manhattan_distances


class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances.

        ```aggregator``` lets you alter how a label is predicted for a data point based 
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to median, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3 
        closest neighbors are:
            [
                [1, 2, 3], 
                [2, 3, 4], 
                [3, 4, 5]
            ] 
        And the aggregator is 'mean', applied along each dimension, this will return for 
        that point:
            [
                [2, 3, 4]
            ]

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean' or 'manhattan'. This is the distance measure
                that will be used to compare features to produce labels. 
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator


    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class 
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional! 
        
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """
        self.train_features = features
        self.train_targets = targets
        

    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the 
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor. 
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """
        # for each row i:
            # Set indexes_list = the n_neighbors nearest of test_features[i] from train_features
            # Set test_target[i] to be  the mean/median of the train_target[j] for j in indexes_list
        predictions = np.zeros((len(features), self.train_targets.shape[1]))
        for i in range(features.shape[0]):
            nearest_neighbors_indexes = self.find_n_nearest_neighbors(self.n_neighbors, features[i], self.train_features, ignore_first)
            predicted_value = self.predict_value(nearest_neighbors_indexes)
            predictions[i] = predicted_value         
        return predictions

    def mode_list(self, l):
        (_, idx, counts) = np.unique(l, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        mode = l[index]
        return mode

    def predict_value(self, nearest_neighbors_indexes):
        label = []
        for j in range(self.train_targets.shape[1]):
            values_list = []
            for i in nearest_neighbors_indexes:
                # train_targets is an array of arrays with size 0
                values_list.append(self.train_targets[i][j])

            if self.aggregator == 'mean':
                label.append(np.mean(values_list, axis=0))
            elif self.aggregator == 'median':
                label.append(np.median(values_list, axis=0))
            else:
                label.append(self.mode_list(values_list))
        return label


    def find_n_nearest_neighbors(self, n_neighbors, feature, train_features, ignore_first):
        if self.distance_measure == 'euclidean':
            dist_list = euclidean_distances(feature, train_features)
        else:
            dist_list = manhattan_distances(feature, train_features)
        return self.find_n_nearest_indexes(n_neighbors, dist_list[0], ignore_first)


    def find_n_nearest_indexes(self, n_neighbors, dist_list, ignore_first):
        if ignore_first:
            dist_list_ignore_first = np.delete(dist_list, dist_list.argmin())
            if len(dist_list_ignore_first) <= n_neighbors:
                neighbors_indexes = range(len(dist_list_ignore_first))
            else:
                neighbors_indexes = np.argpartition(dist_list_ignore_first, n_neighbors)
        else:
            if len(dist_list) <= n_neighbors:
                neighbors_indexes = range(len(dist_list))
            else:
                neighbors_indexes = np.argpartition(dist_list, n_neighbors)
        n_neighbors_indexes = neighbors_indexes[:n_neighbors]
        return n_neighbors_indexes

