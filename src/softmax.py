"""Softmax regression from scratch: core building blocks"""

import numpy as np


def one_hot(
        y: np.ndarray,
    ) -> np.ndarray:
    """One-Hot encoder for y vector.

    Parameters
    ----------
    y : np.ndarray of shape (m,)
        Input array of class labels to encode.

    Returns
    -------
    transformed_matrix : np.ndarray of shape (m, K)
        One-hot encoded representation of y.
    """
    unique_values = len(np.unique(y))
    transformed_matrix = np.zeros((len(y), unique_values))
    transformed_matrix[list(range(len(y))), list(y)] = 1
    return transformed_matrix


def evaluate_gradients_and_loss(
        X: np.ndarray, 
        y: np.ndarray, 
        matrix_weight: np.ndarray,
    ) -> tuple[np.ndarray, float]:
    """Compute gradients and cross-entropy loss for softmax regression.

    Parameters
    ----------
    X : np.ndarray of shape (m, n)
        Input feature matrix, including the bias column.
    y : np.ndarray of shape (m, K)
        One-hot encoded target labels.
    matrix_weight : np.ndarray of shape (K, n)
        Current weight matrix.

    Returns
    -------
    gradients : np.ndarray of shape (K, n)
        Gradient of the cross-entropy loss with respect to matrix_weight.
    loss : float
        Cross-entropy loss value for the given batch.
    """
    scores = X @ matrix_weight.T # scores: (m x K)
    scores = scores - scores.max(axis=1, keepdims=True) # overflow protection
    exp_scores = np.exp(scores)
    proba = exp_scores / exp_scores.sum(axis=1, keepdims=True) # proba: (m x K), one hot y: (m x K)
    loss = cross_entropy(proba, y)
    gradients = ((proba - y).T @ X) / X.shape[0] 
    # (proba - y): (m x K), (proba - y).T: (K x m), X: (m x n)
    # (proba - y).T @ X: (K x n)
    return gradients, loss


def cross_entropy(
        proba: np.ndarray, 
        y: np.ndarray,
    ) -> float:
    """Compute cross-entropy loss.

    Parameters
    ----------
    proba : np.ndarray of shape (m, K)
        Predicted class probabilities.
    y : np.ndarray of shape (m, K)
        One-hot encoded target labels.

    Returns
    -------
    loss : float
        Cross-entropy loss between predicted probabilities and true labels.
    """
    m = y.shape[0]
    return ((y * np.log(proba)).sum(axis=1)).sum(axis=0) / -m


def learning_schedule(
        epoch: int,
        t0: float = 200,
        t1: float = 1000,
    ) -> float:
    """Learning schedule for batch gradient descent.

    Parameters
    ----------
    epoch : int
        Current trainig epoch number.
    t0 : float
        Hyperparameter t0
    t1 : float
        Hyperparameter t1

    Returns
    -------
    learning_rate : float
        Learning rate for the current epoch.
    """
    return t0 / (epoch + t1)
