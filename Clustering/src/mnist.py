
import os
import struct
from array import array as pyarray
import numpy
from numpy import append, array, int8, uint8, zeros
from kmeans import KMeans
from gmm import GMM
import metrics
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


def load_mnist(dataset="training", digits=None, path=None, asbytes=False, selection=None, return_labels=True, return_indices=False):
    """
    Loads MNIST files into a 3D numpy array.

    You have to download the data separately from [MNIST]_. Use the ``path`` parameter
    to specify the directory that contains all four downloaded MNIST files.

    Parameters
    ----------
    dataset : str
        Either "training" or "testing", depending on which dataset you want to
        load.
    digits : list
        Integer list of digits to load. The entire database is loaded if set to
        ``None``. Default is ``None``.
    path : str
        Path to your MNIST datafiles. The default is ``None``, which will try
        to take the path from your environment variable ``MNIST``. The data can
        be downloaded from http://yann.lecun.com/exdb/mnist/.
    asbytes : bool
        If True, returns data as ``numpy.uint8`` in [0, 255] as opposed to
        ``numpy.float64`` in [0.0, 1.0].
    selection : slice
        Using a `slice` object, specify what subset of the dataset to load. An
        example is ``slice(0, 20, 2)``, which would load every other digit
        until--but not including--the twentieth.
    return_labels : bool
        Specify whether or not labels should be returned. This is also a speed
        performance if digits are not specified, since then the labels file
        does not need to be read at all.
    return_indicies : bool
        Specify whether or not to return the MNIST indices that were fetched.
        This is valuable only if digits is specified, because in that case it
        can be valuable to know how far
        in the database it reached.

    Returns
    -------
    images : ndarray
        Image data of shape ``(N, rows, cols)``, where ``N`` is the number of images. If neither labels nor inices are returned, then this is returned directly, and not inside a 1-sized tuple.
    labels : ndarray
        Array of size ``N`` describing the labels. Returned only if ``return_labels`` is `True`, which is default.
    indices : ndarray
        The indices in the database that were returned.

    Examples
    --------
    Assuming that you have downloaded the MNIST database and set the
    environment variable ``$MNIST`` point to the folder, this will load all
    images and labels from the training set:

    >>> images, labels = ag.io.load_mnist('training') # doctest: +SKIP

    Load 100 sevens from the testing set:

    >>> sevens = ag.io.load_mnist('testing', digits=[7], selection=slice(0, 100), return_labels=False) # doctest: +SKIP

    """

    # The files are assumed to have these names and should be found in 'path'
    files = {
        'training': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
        'testing': ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'),
    }

    if path is None:
        try:
            path = os.environ['MNIST']
        except KeyError:
            raise ValueError(
                "Unspecified path requires environment variable $MNIST to be set")

    try:
        images_fname = os.path.join(path, files[dataset][0])
        labels_fname = os.path.join(path, files[dataset][1])
    except KeyError:
        raise ValueError("Data set must be 'testing' or 'training'")

    number_in_test = 100
    # We can skip the labels file only if digits aren't specified and labels aren't asked for
    if return_labels or digits is not None:
        flbl = open(labels_fname, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        labels_raw = pyarray("b", flbl.read())
        flbl.close()

    fimg = open(images_fname, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    images_raw = pyarray("B", fimg.read())
    fimg.close()

    if digits:
        indices = [k for k in range(size) if labels_raw[k] in digits]
    else:
        indices = range(size)

    if selection:
        indices = indices[selection]
    N = len(indices)

    images = zeros((N, rows, cols), dtype=uint8)

    if return_labels:
        labels = zeros((N), dtype=int8)
    for i, index in enumerate(indices):
        images[i] = array(images_raw[indices[i]*rows *
                          cols: (indices[i]+1)*rows*cols]).reshape((rows, cols))
        if return_labels:
            labels[i] = labels_raw[indices[i]]

    if not asbytes:
        images = images.astype(float)/255.0

    ret = (images,)
    if return_labels:
        ret += (labels,)
    if return_indices:
        ret += (indices,)
    if len(ret) == 1:
        return ret[0]  # Don't return a tuple of one
    else:
        return ret


if __name__ == "__main__":
    for n in [6000]:
        images, labels = load_mnist('training', path=".")
        images_train, labels_train = load_mnist(
            'testing', path=".", selection=slice(0, n))
        images_test, labels_test = load_mnist(
            'testing', path=".", selection=slice(n, n*2+2))
        # Fit & predict GMM
        # Evaluate GMM
        # Fit & Predict KM
        # Eval KMeans
        # Compare - not with code

        # question 3:
        flatten_train_array = []
        flatten_test_array = []

        for i in range(images_train.shape[0]):
            flatten_train_array.append(images_train[i].flatten())
        flatten_train_array = numpy.array(flatten_train_array)

        for i in range(images_test.shape[0]):
            flatten_test_array.append(images_test[i].flatten())
        flatten_test_array = numpy.array(flatten_test_array)

        n_clusters = 10
        # kmeans = KMeans(n_clusters)
        # gmm = GMM(n_clusters, covariance_type='spherical')
        kmeans = KMeans(n_clusters=n_clusters)
        gmm = GaussianMixture(n_components=n_clusters)

        # kmeans.fit(flatten_train_array)
        # gmm.fit(flatten_train_array)
        kmeans.fit(flatten_train_array)
        gmm.fit(flatten_train_array)

        # kmeans_prediction = kmeans.predict(flatten_test_array)
        # gmm_prediction = gmm.predict(flatten_test_array)
        kmeans_prediction = kmeans.predict(flatten_test_array)
        gmm_prediction = gmm.predict(flatten_test_array)

        test_kmeans_prediction = metrics.adjusted_mutual_info(
            labels_test, kmeans_prediction)
        test_gmm_prediction = metrics.adjusted_mutual_info(
            labels_test, gmm_prediction)

        print("test_kmeans_prediction: ", test_kmeans_prediction,
              images_train.shape, images_test.shape)
        print("test_gmm_prediction: ", test_gmm_prediction,
              images_train.shape, images_test.shape)
        pass

        # question 4:
        from statistics import mode

        for prediction in [kmeans_prediction, gmm_prediction]:
            clusters_index_matrix = numpy.empty((n_clusters,), dtype=object)
            clusters_label_matrix = numpy.empty((n_clusters,), dtype=object)
            for i in range(len(clusters_index_matrix)):
                clusters_index_matrix[i] = []
                clusters_label_matrix[i] = []
            for i in range(len(prediction)):
                clusters_index_matrix[prediction[i]].append(i)
                clusters_label_matrix[prediction[i]].append(labels_test[i])

            equal_to_cluster = 0
            for row in range(clusters_label_matrix.shape[0]):
                if len(clusters_label_matrix[row]) == 0:
                    continue
                common_label = mode(clusters_label_matrix[row])
                equal_to_cluster += clusters_label_matrix[row].count(
                    common_label)

            accuracy = equal_to_cluster/prediction.shape[0]
            print("accuracy: ", accuracy)
        pass

        # question 4:
        from matplotlib import pyplot as plt

        fig = plt.figure()
        for i in range(kmeans.cluster_centers_.shape[0]):
            reshapes_cluster_centers = (
                kmeans.cluster_centers_[i]).reshape((28, 28))

            dist_list = []
            for j in range(flatten_test_array.shape[0]):
                dist_list.append(numpy.linalg.norm(
                    flatten_test_array[j] - kmeans.cluster_centers_[i]))
            min_index = dist_list.index(min(dist_list))
            closest_image = images_test[min_index]

            sub = fig.add_subplot(10, 2, 2*i + 1)
            sub.imshow(reshapes_cluster_centers, interpolation='nearest')
            if i == 0:
                sub.set_title('The Mean')

            sub = fig.add_subplot(10, 2, 2*i + 2)
            sub.imshow(closest_image, interpolation='nearest')
            if i == 0:
                sub.set_title('The nearest example')

        plt.show()
