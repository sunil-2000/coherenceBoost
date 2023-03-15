import torch
from transformers import AutoTokenizer, GPT2LMHeadModel


class CB:
    """
    coherence-boosted GPT model
    """

    def __init__(self, alpha, k, model_id="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = GPT2LMHeadModel.from_pretrained(model_id)
        self.alpha = alpha
        self.k = k

    def generate_w_boost(self, x, fmax=False, decode=False):
        """
        single example from two models
        argmax(P[w_n| w_0...w_n-1] + P[w_n| w_k...w_n-1])

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

    def set_attn_mask(self):
        """
        set attention mask to for gpt inference with batches
        """
        raise NotImplemented
