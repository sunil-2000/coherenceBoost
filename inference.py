# example script to generate boosted predictions
from datasets import load_dataset
from cb import CB
from util import extract_label

# load dataset
lambada = load_dataset("lambada")
test_set = lambada["test"].map(extract_label)
X, Y = test_set["feat"], test_set["label"]

# load boosted model
alpha = -0.6
k = 10

cb_model = CB(alpha, k, device="cpu")
y_pred_boosted = cb_model.boosted_batched_generate(X)
