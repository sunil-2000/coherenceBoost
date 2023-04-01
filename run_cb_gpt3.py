from datasets import load_dataset
from cb_gpt3 import CBGPT3
from util import eval_lambada

# load dataset

lambada = load_dataset("EleutherAI/lambada_openai")
X = lambada["test"]["text"]

alpha, k, model = -0.3, 10, 'babbage'
file_name = f"{model}_preds.csv"

cb = CBGPT3(alpha, k, model=model)
print(model)
cb.generate(X, file_name, batch_size=128)

eval_lambada(file_name)