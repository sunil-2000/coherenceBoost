import pandas as pd

def accuracy(y, y_pred):
    """
    y: 1D tensor
    y_pred: 1D tensor
    """
    return (y == y_pred).sum() / len(y_pred)


def eval_lambada(file_name):
  df = pd.read_csv(file_name)
  fx_preds = df['fx_token'].to_numpy()
  cb_preds = df['cb_token'].to_numpy()
  cb1_preds = df['cb_token1'].to_numpy()
  s_preds = df['s_token'].to_numpy
  targets = df['target'].to_numpy()

  print(fx_preds.shape, cb_preds.shape, targets.shape)
  print(accuracy(fx_preds, targets))
  print(accuracy(cb_preds, targets))
  print(accuracy(cb1_preds, targets))
  print(accuracy(s_preds, targets))

