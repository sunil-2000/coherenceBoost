import openai
import time
import csv
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
from private import api_key

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CBGPT3:
    """
    coherence boosted gpt-3
    """

    def __init__(self, alpha, k, model="ada"):
        openai.api_key = api_key
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.alpha = alpha
        self.k = k
        self.model = model

    def load_data(self, encoded_inputs, batch_size):
        """
        return dataloader and dataset of input
        """

        def ctx_target_split(example):
            """
            paper states they evaluate LAMBADA on last token prediction
            """
            example["ctx"] = example["input_ids"][:-1]
            example["target"] = example["input_ids"][-1]
            return example

        # generate dataset to process in batches
        dataset = Dataset.from_dict(encoded_inputs)
        dataset = dataset.map(ctx_target_split)
        dataset.set_format(
            type="torch", columns=["ctx", "target", "input_ids", "attention_mask"]
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader, dataset

    def generate(self, X, file_name, batch_size=32):
        """
        X: passage list
        file_name: file to write results to
        """
        encoded_inputs = self.tokenizer(X, padding=True)
        dataloader, _ = self.load_data(encoded_inputs, batch_size)

        with open(file_name, "w") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "id",
                    "model",
                    "fx_token",
                    "cb_token",
                    "cb_token1",
                    "s_token",
                    "target",
                ]
            )

        for i, data in enumerate(tqdm(dataloader)):
            prompt_lst, short_prompt_lst = [], []
            ctx = data["ctx"]
            mask = data["attention_mask"][:, :-1]
            targets = data["target"].tolist()

            # eos appended as padding produces different outputs with tests, so squeeze each tensor
            for j in range(ctx.shape[0]):
                prompt = ctx[j] * mask[j]
                prompt = prompt[prompt.nonzero()].flatten().tolist()
                prompt_lst.append(prompt)
                short_prompt_lst.append(prompt[-self.k :])

            # run models
            res_full = openai.Completion.create(  # full context
                model=self.model,
                prompt=prompt_lst,
                logprobs=5,  # max amount of logprobs
                max_tokens=1,
                temperature=0,
            )
            res_short = openai.Completion.create(  # short context
                model=self.model,
                prompt=short_prompt_lst,
                logprobs=5,  # max amount of logprobs
                max_tokens=1,
                temperature=0,
            )

            # save data
            rows_to_save = self.extract_preds(
                res_full, res_short, targets, i * batch_size
            )
            with open(file_name, "a") as f:
                w = csv.writer(f)
                w.writerows(rows_to_save)

            time.sleep(20)

    def extract_preds(self, res_full, res_short, targets, id_start):
        rows = []
        for i, (out_full, out_short) in enumerate(
            zip(res_full["choices"], res_short["choices"])
        ):
            logits = list(out_full["logprobs"]["top_logprobs"][0].values())
            words = list(out_full["logprobs"]["top_logprobs"][0].keys())
            tokens = [self.tokenizer(w)["input_ids"][0] for w in words]
            fx_token = tokens[np.argmax(logits)]

            # short preds
            s_logits = list(out_short["logprobs"]["top_logprobs"][0].values())
            s_words = list(out_short["logprobs"]["top_logprobs"][0].keys())
            s_tokens = [self.tokenizer(w)["input_ids"][0] for w in s_words]

            unk_logprob = np.log(
                1
                - np.sum([np.exp(x) for x in s_logits])
                / (self.tokenizer.vocab_size - 5)
            )
            cb_logits = [l for l in logits]
            cb_logits1 = [l for l in logits]
            for j, token in enumerate(tokens):
                if token in s_tokens:
                    cb_logits[j] = cb_logits[j] - self.alpha * s_logits[j]
                    cb_logits1[j] = cb_logits1[j] - self.alpha * s_logits[j]
                else:
                    cb_logits[j] = cb_logits[j] + self.alpha * unk_logprob

            cb_token = tokens[np.argmax(cb_logits)]
            cb_token1 = tokens[np.argmax(cb_logits1)]
            s_token = s_tokens[np.argmax(s_logits)]

            row = {
                "id": id_start + i,
                "model": self.model,
                "fx_token": fx_token,
                "cb_token": cb_token,
                "cb_token1": cb_token1,
                "s_token": s_token,
                "target": targets[i],
            }
            rows.append(list(row.values()))

        return rows
