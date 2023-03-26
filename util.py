def accuracy(y, y_pred):
    """
    y: 1D tensor
    y_pred: 1D tensor
    """
    return (y == y_pred).sum() / len(y_pred)
