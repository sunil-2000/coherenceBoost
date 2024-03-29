import torch
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel


class CB:
    """
    coherence-boosted GPT model
    """

    def __init__(self, alpha, k, model_id="gpt2", device="cuda"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_id)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.to(self.device)
        self.alpha = alpha
        self.k = k

    def generate_w_boost(self, x, fmax=False, decode=False):
        """
        single example from two models
        `argmax(P[w_n| w_0...w_n-1] + alpha * P[w_n| w_k...w_n-1])`

        fmax(bool): whether to return og model output as well
        decode(bool): output token or word representation
        """
        encoded_x = self.tokenizer(x, return_tensors="pt")["input_ids"]
        encoded_short_x = encoded_x[0][-self.k :][None, :]
        f_x = self.model.generate(
            encoded_x["input_ids"], output_scores=True, return_dict_in_generate=True
        )["scores"]
        f_k = self.model.generate(
            encoded_short_x,
            output_scores=True,
            return_dict_in_generate=True,
        )["scores"]

        out_og = torch.argmax(f_x[0])
        out_boost = torch.argmax(f_x[0] + self.alpha * f_k[0])  # boosted

        if not decode:
            return out_boost if not fmax else out_boost, out_og
        else:
            if not fmax:
                return self.tokenizer(out_boost)
            else:
                return self.tokenizer(out_boost), self.tokenizr(out_og)

    def generate_next_word(self, x):
        """
        single example, next word prediction
        P[w_n | w_0...w_n-1]
        """
        encoded_x = self.tokenizer(x, return_tensors="pt")
        output = self.model.generate(
            encoded_x["input_ids"], output_scores=True, return_dict_in_generate=True
        )
        return self.tokenizer.decode(output["scores"][0].argmax())

    def load_data(self, encoded_inputs, batch_size):
        """
        return dataloader and dataset of input
        """

        def ctx_target_split(example):
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

    def batched_generate(self, X, batch_size=32):
        """
        generate batched predictions and return generated token ids
        X: 2D list of passages
        returns dict {preds, targets}
        """
        encoded_inputs = self.tokenizer(X, padding=True, return_tensors="pt")
        encoded_inputs.to(self.device)
        dataloader, dataset = self.load_data(encoded_inputs, batch_size)

        # run predictions in batches
        preds = []
        for data in tqdm(dataloader):
            input_ids = data["ctx"].to(self.device)
            mask = data["attention_mask"].to(self.device)
            out = self.model.generate(
                input_ids,
                attention_mask=mask[:, :-1],
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )["scores"][0]
            pred_token = torch.argmax(out, axis=1).to("cpu")
            pred_token = pred_token.tolist()
            preds.extend(pred_token)

        return {"preds": torch.tensor(preds), "targets": dataset["target"]}

    def boosted_batched_generate(self, X, batch_size=32, fmax_score=False):
        """
        boosted batched generation
        fmax_score(bool): output boosted and non-boosted predictions
        """
        encoded_inputs = self.tokenizer(X, padding=True, return_tensors="pt")
        encoded_inputs.to(self.device)
        dataloader, dataset = self.load_data(encoded_inputs, batch_size)

        # run predictions in batches
        preds_fmax = []
        preds = []
        for data in tqdm(dataloader):
            input_ids = data["ctx"].to(self.device)
            short_input_ids = data["ctx"][:, -self.k :].to(self.device)
            mask = data["attention_mask"].to(self.device)

            out = self.model.generate(
                input_ids,
                attention_mask=mask[:, :-1],
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )["scores"][0]
            out_k = self.model.generate(
                short_input_ids,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )["scores"][0]

            boosted_score = out + self.alpha * out_k
            pred_token = torch.argmax(boosted_score, axis=1).to("cpu")
            pred_token = pred_token.tolist()
            preds.extend(pred_token)

            if fmax_score:
                pred_token = torch.argmax(out, axis=1).to("cpu")
                pred_token = pred_token.tolist()
                preds_fmax.extend(pred_token)

        out = {"preds_cb": torch.tensor(preds), "targets": dataset["target"]}
        if fmax_score:
            out["preds_fmax"] = torch.tensor(preds_fmax)
        return out

    def tokenize_label(self, Y, rev=False):
        """
        convert true label to their token representation
        Y: 1D list of target labels (str)
        rev(bool): whether to return first or last subtoken of label
        """
        idx = 0 if not rev else -1
        return [self.tokenizer(i)["input_ids"][idx] for i in Y]
