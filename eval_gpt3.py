import os
import csv
import pandas as pd
from util import accuracy

csv_dir = 'gpt3_preds'
params = {'ada':[-0.3, 9], 'babbage':[-0.3, 10], 'curie': [-0.3, 3], 'davinci': [-0.2, 3]}


header = ['model', 'fmax_acc', 'cb_acc', 'alpha', 'k']
accuracies = []
for f in os.listdir(csv_dir):
    model = f.split('_')[0]
    alpha, k = params[model]

    df = pd.read_csv(f"{csv_dir}/{f}")
    fmax = df['fx_token'].to_numpy('float')
    cb = df['cb_token'].to_numpy('float')
    targets = df['target'].to_numpy('float')
    fmax_acc = accuracy(fmax, targets)
    cb_acc = accuracy(cb, targets)
    accuracies.append([model, fmax_acc, cb_acc, alpha, k])

with open('gpt3_fmax_cb_accs.csv', "w") as f:
  w = csv.writer(f)
  w.writerow(header)
  w.writerows(accuracies)


