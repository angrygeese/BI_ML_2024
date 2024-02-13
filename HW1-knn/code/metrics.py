import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    confusion_matrix = np.matrix("'TP', 'FN'; 'FP', 'TN'")
    confusion_dict = dict.fromkeys(confusion_matrix.flat, 0)
    y_pred_int, y_true_int = np.array(list(map(int, y_pred))), np.array(list(map(int, y_true)))
    for predicted_y, test_y in zip(y_pred_int, y_true_int):
        event_type = confusion_matrix[predicted_y, test_y]
        confusion_dict[event_type] += 1

    precision = confusion_dict['TP'] / (confusion_dict['TP'] + confusion_dict['FP'])
    recall = confusion_dict['TP'] / (confusion_dict['TP'] + confusion_dict['FN'])
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (confusion_dict['TP'] + confusion_dict['TN']) / np.array(list(confusion_dict.values())).sum()

    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    y_pred_int, y_true_int = np.array(list(map(int, y_pred))), np.array(list(map(int, y_true)))
    accuracy = np.sum(y_pred_int == y_true_int) / len(y_pred_int)

    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    y_mean = np.mean(y_true)
    sst = np.sum((y_true - y_mean) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (sse / sst)

    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    error = y_pred - y_true
    sq_error = error ** 2
    mse = np.mean(sq_error)

    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    abs_error = np.abs(y_pred - y_true)
    mae = np.mean(abs_error)

    return mae
    