# example script to generate boosted predictions
import csv
from tqdm import tqdm
from datasets import load_dataset
from cb import CB
from util import accuracy

# load dataset
lambada = load_dataset("EleutherAI/lambada_openai")
X = lambada["test"]["text"]

# boosting params
alphas = [-0.6, -0.5, -0.5, -0.5]
ks = [10, 11, 10, 9]
models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

accs = []
for i, model in enumerate(tqdm(models)):
    alpha, k = alphas[i], ks[i]
    cb_model = CB(alpha, k, model_id=model, device="cuda")
    out = cb_model.boosted_batched_generate(X, fmax_score=True, batch_size=64)
    acc = accuracy(out["targets"], out["preds_fmax"])
    acc_boost = accuracy(out["targets"], out["preds_cb"])
    accs.append({"model": model, "fmax": acc, "cb": acc_boost})

file_name = "boosted_lambada.csv"
with open(file_name, "w") as f:
    w = csv.DictWriter(f, accs[0].keys())
    w.writeheader()
    for row in accs:
        values = [i[1] for i in row.items()][1:]
        out = {"model": row["model"], "fmax": values[0].item(), "cb": values[1].item()}
        w.writerow(out)
