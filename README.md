# coherence-boosted GPT

Replication project for [Coherence boosting: When your pretrained language model is not paying enough attention](https://arxiv.org/pdf/2110.08294.pdf)

I attempt to reproduce exact results of GPT2 and the coherence boosted GPT2
evaluation on LAMBADA; this is from column 1 of Table 1 in the paper.

![Table1](table.png)

The authors of the paper state, "We perform experiments with the GPT family of
models, closely replicating the evaluation setting of Radford et al. (2019)." As such,
I evaluate GPT2 similar to how OpenAI evaluated their model. However, I quickly
discovered that their evaluation setting for GPT is not best documented. [Others](https://github.com/openai/gpt-2/issues/131) have expressed confusion over how OpenAI evaluated GPT.
First, OpenAI pre-processed the LAMBADA dataset and evaluated GPT2 on this
[pre-processed dataset](https://huggingface.co/datasets/EleutherAI/lambada_openai).
I found this out as evaluation on the standard [LAMBADA dataset](https://huggingface.co/datasets/lambada) yielded strictly worse results compared to the processed LAMBADA dataset.

Because it is not entirely clear how openAI or the authors of the paper evaluated
GPT, I tried 5 different evaluation accuracies and share the results below and
in the jupyter notebook (demo.ipynb). While, the accuracy results do not match
the table exactly from the notebook, each result shows that coherence boosting
improves performance on LAMBADA significantly. This aligns with the findings
from the paper. The results that I believe most closely follow the experimental 
set-up are computing accuracy of predicting the last sub-token with beam search (beam_width=5). I use a beam width of 5 because this is how OpenAI seems to have evaluated GPT2 (https://github.com/openai/gpt-2/issues/131). GPT2-small achieves an accuracy of 46.1% and the boosted GPT2-small achieves an accuracy of 60.4%; the paper achieved 47.66% and 66.70% respectively. Below are all the accuracies GPT2-small achieved under different accuracy criterions.

|  GPT2  (124M)                                |   Results      | Paper Results |
| ---                                    | ----           | ---           |
|  $f_{max}$                             |  46.1%         |  47.66%       |
|  CB ($\alpha_{k} = \alpha^{*}_{k} $)   |  60.4%         |  66.70%       |
|  $\alpha^{*}_{k}$                      |  -0.6          |  -0.6         |
|  $k^{*}$                               |  10            |  10           |

## Accuracy evaluations definitions:
- last word accuracy (`lw_acc`): whether predicted token matches true last word token

- first and last subtoken of last word accuracy (`ft_acc`, `lt_acc`): whether first predicted subtoken matches first subtoken of true last word; whether last predicted subtoken matches first subtoken of true last word.

- first and last subtoken of last word accuracy with beam search (`ft_beam_acc`, `lt_beam_acc`): same as above with
beam width of 5. 

## Full accuracy evaluation results:
where $\alpha^{*}_{k}=-0.6, k^{*}=10$

| GPT2  (124M)                              | lw_acc | ft_acc | lt_acc | ft_beam_acc | lt_beam_acc |
| ---                                 | ---    | ---    | ---    | ---         |    ---      |
| $f_{max}$                           | 25.07% | 32.84% | 25.07% | 60.37%      |  46.13%     |
| CB ($\alpha_{k} = \alpha^{*}_{k} $) | 43.70% | 55.81% | 43.70% | 78.27%      |  60.41%     |
