import torch
from torch.nn import functional as F
from torchmetrics import Accuracy, F1Score
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy, multilabel_accuracy
from torch.nn.utils.rnn import pad_sequence
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embedding_lookup(word_embedding, x):
    '''
    input(s) shape: [num_words, embedding_dim], [batch_size, doc_len, sen_len]
    output shape: [batch_size, doc_len, sen_len, embedding_dim]
    '''
    x = F.embedding(torch.from_numpy(x).type(torch.LongTensor), word_embedding)
    # x = F.embedding(x.type(torch.LongTensor), word_embedding)
    return x


def f1_acc(pred_y, true_y, dataset):
    true_y = true_y.to('cpu')
    pred_y = pred_y.to('cpu')
    _, true_indices = torch.max(true_y, 1)
    _, pred_indices = torch.max(pred_y, 1)
    if dataset == "ECF":
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


def bert_accuracy(pred_y, true_y):
    true_indices = true_y.to('cpu')
    pred_y = pred_y.to('cpu')
    _, pred_indices = torch.max(pred_y, 1)
    f1 = multiclass_f1_score(pred_indices, true_indices, num_classes=7, average="macro")
    acc = multiclass_accuracy(pred_indices, true_indices, num_classes=7, average="macro")
    accuracy = Accuracy(task="multiclass", num_classes=7)
    acc2 = accuracy(pred_indices, true_indices)
    f1macro = F1Score(task="multiclass", num_classes=7, average="macro")
    f1scuff = f1macro(pred_indices, true_indices)
    return f1, acc2


def batch_index(length, batch_size, test=False):
    index = list(range(length))
    if test == False:
        np.random.shuffle(index)
    for i in range(int((length + batch_size - 1) / batch_size)):
        ret = index[i * batch_size: (i + 1) * batch_size]
        if test == False and len(ret) < batch_size: break
        yield ret


def get_batch_data_pair(x, y, batch_size, test=False):
    for index in batch_index(len(y), batch_size, test):
        x_batch = [torch.from_numpy(x[i]) for i in index]
        padded_x_batch = pad_sequence(x_batch, batch_first=True)
        # padded_to = list(padded_x_batch.size())[1]
        # padded_batch = padded_x_batch.reshape(len(padded_x_batch), padded_to, 1)
        feed_list = [padded_x_batch, y[index]]
        yield feed_list, len(index)


def get_batch_data_pair_old(x, y, batch_size, test=False):
    for index in batch_index(len(y), batch_size, test):
        feed_list = [x[index], y[index]]
        yield feed_list, len(index)
