import numpy as np


def get_true_shapes(X):
    try:
        shape = (X.shape[0], X.shape[1])
    except IndexError:
        shape = (1, X.shape[0])
    return shape


def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    X_shape = get_true_shapes(X)
    Y_shape = get_true_shapes(Y)
    D = np.zeros((X_shape[0], Y_shape[0]))
    for i in range(X_shape[0]):
        for j in range(Y_shape[0]):
            if type(X[i]) == np.ndarray:
                euclidean_dist = np.linalg.norm(X[i] - Y[j])
            else:
                euclidean_dist = np.linalg.norm(X - Y[j])
            D[i][j] = euclidean_dist
    return D


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    # D = np.zeros((X.shape[0], Y.shape[0]))
    # for i in range(X.shape[0]):
    #     for j in range(Y.shape[0]):
    #         D[i][j] = np.sum([abs(k-l) for k, l in zip(X[i], Y[j])])
    # return D

    X_shape = get_true_shapes(X)
    Y_shape = get_true_shapes(Y)
    D = np.zeros((X_shape[0], Y_shape[0]))
    for i in range(X_shape[0]):
        for j in range(Y_shape[0]):
            if type(X[i]) == np.ndarray:
                D[i][j] = np.sum([abs(k-l) for k, l in zip(X[i], Y[j])])
            else:
                D[i][j] = np.sum([abs(k-l) for k, l in zip(X, Y[j])])
    return D
