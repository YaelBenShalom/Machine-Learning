import numpy as np
import copy


def choose_random_means(n_clusters, features):
    mean_list = features[np.random.choice(
        features.shape[0], n_clusters, replace=False)]
    return mean_list


def get_mean_index(mean_list, feature):
    dist_list = []
    for i in range(len(mean_list)):
        dist_list.append(np.linalg.norm(mean_list[i] - feature))
    mean_index = dist_list.index(min(dist_list))
    return mean_index


def get_new_mean_list(label_matrix):
    new_mean_list = np.zeros((len(label_matrix), len(label_matrix[0][0])))
    for i in range(len(label_matrix)):
        new_mean_list[i] = (np.mean(label_matrix[i], axis=0))
    return new_mean_list


class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        label_matrix = np.empty((self.n_clusters, ), dtype=object)
        for i in range(len(label_matrix)):
            label_matrix[i] = []
        old_mean_list = None
        new_mean_list = choose_random_means(self.n_clusters, features)

        while not (np.array(new_mean_list) == np.array(old_mean_list)).all():
            old_mean_list = copy.deepcopy(new_mean_list)
            for i in range(len(label_matrix)):
                label_matrix[i] = []
            for f in features:
                mean_index = get_mean_index(old_mean_list, f)
                label_matrix[mean_index].append(f)
            new_mean_list = get_new_mean_list(label_matrix)
        self.means = new_mean_list

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        prediction = np.empty((features.shape[0], ), dtype=object)
        for i in range(len(features)):
            mean_index = get_mean_index(self.means, features[i])
            prediction[i] = mean_index
        return prediction
