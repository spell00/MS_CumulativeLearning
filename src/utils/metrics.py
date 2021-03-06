# Libraries
import numpy as np
from tensorflow.keras import backend as K

smooth = 1


def jaccard_distance_loss(y_true, y_pred, smooth=100, backend=K):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(backend.abs(y_true_f * y_pred_f))
    sum_ = backend.sum(backend.abs(y_true_f) + backend.abs(y_pred_f))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def mean_length_error(y_true, y_pred, backend=K):
    y_true_f = backend.sum(backend.round(backend.flatten(y_true)))
    y_pred_f = backend.sum(backend.round(backend.flatten(y_pred)))
    delta = (y_pred_f - y_true_f)
    return backend.mean(backend.tanh(delta))


def dice_coef(y_true, y_pred, backend=K):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred, backend=K):
    return -dice_coef(y_true, y_pred, backend=backend)


def np_dice_coef(y_true, y_pred):
    tr = y_true.flatten()
    pr = y_pred.flatten()
    return (2. * np.sum(tr * pr) + smooth) / (np.sum(tr) + np.sum(pr) + smooth)


# matthews_correlation
def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
