import torch
from torch.nn import functional as F
from torchmetrics import Accuracy, F1Score
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy, multilabel_accuracy
from torch.nn.utils.rnn import pad_sequence
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embedding_lookup(word_embedding, x):
    '''
    Method is used to transform embedding id vector into its embeddings.
    input(s) shape: [num_words, embedding_dim], [batch_size, doc_len, sen_len]
    output shape: [batch_size, doc_len, sen_len, embedding_dim]
    '''
    x = F.embedding(torch.from_numpy(x).type(torch.LongTensor), word_embedding)
    # x = F.embedding(x.type(torch.LongTensor), word_embedding)
    return x


def f1_acc(pred_y, true_y, dataset):
    '''
    Returns accuracy and f1 score of the input predictions with the ground truth
    '''
    true_y = true_y.to('cpu')
    pred_y = pred_y.to('cpu')
    _, true_indices = torch.max(true_y, 1)
    _, pred_indices = torch.max(pred_y, 1)
    if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
        f1 = multiclass_f1_score(pred_indices, true_indices, num_classes=7, average="macro")
        accuracy = Accuracy(task="multiclass", num_classes=7)
    elif dataset == "RAVDESS":
        f1 = multiclass_f1_score(pred_indices, true_indices, num_classes=8, average="macro")
        accuracy = Accuracy(task="multiclass", num_classes=8)
    else:
        f1 = multiclass_f1_score(pred_indices, true_indices, num_classes=5, average="macro")
        accuracy = Accuracy(task="multiclass", num_classes=5)
    acc2 = accuracy(pred_indices, true_indices)
    return f1, acc2
