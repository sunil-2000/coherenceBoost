import torch
from datasets import Dataset
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import AutoTokenizer, GPT2LMHeadModel

class CB:
    """
    coherence-boosted GPT model
    """

    def __init__(self, alpha, k, model_id="gpt2", device='cuda'):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left" 
        self.model = GPT2LMHeadModel.from_pretrained(model_id)
        self.model.to(self.device)
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

    def batched_generate(self, X, batch_size=32):
        """
        generate batched predictions and return generated token ids
        X: 2D list of passages 
        """
        encoded_inputs = self.tokenizer(X, padding=True, return_tensors="pt")
        encoded_inputs.to(self.device)
        # generate dataset to process in batches
        dataset = Dataset.from_dict(encoded_inputs)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # run predictions in batches
        # also track scores (so don't need to run model twice for boosted preds)
        preds = []
        for i, data in enumerate(dataloader):
            print(f"processing batch {i}")
            input_ids, mask = data['input_ids'].to(self.device), data['attention_mask'].to(self.device)
            out = self.model.generate(input_ids, attention_mask=mask, output_scores=True, return_dict_in_generate=True, max_new_tokens=1, pad_token_id=self.tokenizer.eos_token_id)['scores'][0]
            pred_token = torch.argmax(out, axis=1)
            preds.append(pred_token)
            print("-"*50)

        # flatten preds
        flat_preds = [pred for batch in preds for pred in batch]
        return flat_preds

    def k_batched_generate(self, X):
        """
        batched generation with shorter context
        """
        raise NotImplementedError
    
    def boosted_batched_generate(self, X, batch_size=32):
        """
        generate boosted batched predictions and return generated token ids
        X: 2D list of passages
        """
        raise NotImplementedError
        encoded_input = self.tokenizer(X, padding=True, return_tensors="pt")
        full_mask = encoded_input['attention_mask']
        n, d = encoded_input['input_ids'].shape
        print(f'len(X): {len(X)}, encoded_input shape: {encoded_input.shape}')
        # for i in encoded_input:
        #     print(i[-self.k:][None, :])
        #     0/0
        encoded_short = torch.stack([i[-self.k :] for i in encoded_input])
        # attend to all but last 10 positions
        short_mask = torch.zeros(d)
        short_mask[-self.k:] = 1
        print(f'encoded_short shape: {encoded_short.shape}')

        # full_input = torch.stack((encoded_input, full_mask), axis=1).to(self.device)
        # short_input = torch.stack((encoded_short, short_mask), axis=1).to(self.device)
        # encoded_input = encoded_input.to(self.device)
        # encoded_short = encoded_short.to(self.device)


        # generate dataset to process in batches
        dataset = TensorDataset(encoded_input)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # run predictions in batches
        max_preds = []
        boosted_preds = []
        for i, (data_max) in enumerate(dataloader):
            print(f"processing batch {i}")
            out_max = self.model.generate(data_max, output_scores=True, return_dict_in_generate=True, max_new_tokens=1)['scores'][0]
            print(data_max)
            0/0
            out_k = self.model.generate(data_k, output_scores=True, return_dict_in_generate=True, max_new_tokens=1)['scores'][0]
            
            # stack out, k
            out_boost = torch.argmax(out_max + (self.alpha * out_k), axis=1)
            out_full_ctx = torch.argmax(out_max, axis=1)
            
            boosted_preds.append(out_boost)
            max_preds.append(out_full_ctx)
            print("-"*50)

        # flatten _preds
        return max_preds, boosted_preds
    
    def generate_dataset(self):
        pass

    def label_to_token(self, Y):
        """
        convert true label to their token representation
        Y: 1D list of target labels
        """
        return self.tokenizer(Y)['inputed_ids']
