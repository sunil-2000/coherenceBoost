def extract_label(datum):
    """
    fn to apply over lambada dataset for feature,label extraction
    """
    text_lst = datum["text"].split(" ")
    datum["feat"] = " ".join(text_lst[:-1])
    datum["label"] = " " + text_lst[-1]
    return datum


def accuracy(y_pred, y_true):
    """
    accuracy metric for lambada

    y_pred: 1D tensor
    y_true: 1D tensor
    """
    return (y_pred == y_true).sum() / len(y_pred)


def accuracy_last_word(model, y_pred, y_true):
    """
    accuracy of predicting last word
    """
    y_pred = [model.tokenizer.decode(pred).strip() for pred in y_pred]

    acc = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i].strip():
            acc += 1

    return acc / len(y_pred)


def first_token_acc_aux(model, y_pred, y_true):
    y_true_tokens = [model.tokenizer(i)["input_ids"] for i in y_true]

    for i, j in enumerate(y_true_tokens):
        if len(j) > 1:
            print(f"full actual Y: {y_true[i]}")
            print(f"first token of actual: {model.tokenizer.decode(j[0])}")
            print(f"second token of actual: {model.tokenizer.decode(j[1])}")
            print(f"y_pred == actual first token: {y_pred[i] == j[0]}")
            print(f"y_pred: {model.tokenizer.decode(y_pred[i])}")
            print("\n")


def first_token_acc(y_pred, y):
    """
    y_pred: list of predicted tokens
    y: list of true tokens (first token of true word)
    """
    acc = 0
    for i, y_true_token in enumerate(y):
        if y_pred[i] == y_true_token:
            acc += 1

    return acc / len(y)
