import os
import torch
import numpy as np
import vectors_creation
from models import LSTM_extractor, LSTM_extractor_old, ConvNet, CNN_small
from dataset_loading import load_dataset, load_old_dataset, load_RAVDESS
import soundfile as sf
import norbert
import nussl
import torcheval.metrics.functional
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from torch.nn import functional as F
import matplotlib.pyplot as plt
import librosa.display
from torch.nn.utils.rnn import pad_sequence
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scuff_accuracy(pred_y, true_y):
    true_y = true_y.to('cpu')
    pred_y = pred_y.to('cpu')
    _, true_indices = torch.max(true_y, 1)
    _, pred_indices = torch.max(pred_y, 1)
    f1 = multiclass_f1_score(pred_indices, true_indices, num_classes=8, average="macro")
    acc = multiclass_accuracy(pred_indices, true_indices, num_classes=8, average="macro")
    return f1, acc


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


def main(config):
    model = CNN_small()
    # model = ConvNet()
    train_x, val_x, train_y, val_y, train_speakers, val_speakers, test_x, test_y, test_speakers = load_dataset()
    # val_x = val_x[:, None, :]
    # test_x = test_x[:, None, :]
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0.0001)
    loss_f = torch.nn.CrossEntropyLoss()
    # wandb.init(project="DP", entity="zeman339", config=config)
    for epoch in range(1, 50):
        counter = 0
        for train, _ in get_batch_data_pair(train_x, train_y, config["batch_size"]):
            tr_x_batch, tr_y_batch = train
            model.train()
            optimizer.zero_grad()
            # tr_x_batch = tr_x_batch[:, None, :]
            tr_x_batch = tr_x_batch.float()
            tr_x_batch = tr_x_batch[:, None, :, :]
            tr_pred_y, _ = model(tr_x_batch.to(device))
            # tr_pred_y = torch.squeeze(tr_pred_y)
            tr_y_batch_f = tr_y_batch.float()
            loss = loss_f(tr_y_batch_f.to(device), tr_pred_y.float())
            counter += 1
            if counter % 20 == 0:
                model.eval()
                # vals = []
                eval_pred_y, _ = model(tr_x_batch.to(device))
                # eval_pred_y = torch.squeeze(eval_pred_y)
                # for value in val_x:
                #     value = value[None, :, :]
                #     val_pred_y, _ = model(value.to(device))
                #     val_pred_y = torch.squeeze(val_pred_y)
                #     vals.append(val_pred_y)
                # val_pred_y = torch.stack(vals, dim=0)
                # f1, acc = scuff_accuracy(val_pred_y, val_y)
                # # wandb.log({"test_acc": acc, "test_f1": f1})
                # print("epoch: " + str(epoch) + "\nacc: " + str(acc) + "\nf1: " + str(f1) + "\n")
                f1, acc = scuff_accuracy(eval_pred_y, tr_y_batch_f)
                # wandb.log({"train_acc": acc, "train_f1": f1})
                print("train acc: " + str(acc) + ";  train f1: " + str(f1) + ";  loss: " + str(loss))

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            tests = []
            tests_y = []
            test_pred_y, _ = model(test_x.to(device))
            # test_pred_y = torch.squeeze(test_pred_y)
            # for value in test_x:
            #     value = value[None, :, :]
            #     value = torch.from_numpy(value)
            #     test_pred_y, _ = model(value.to(device))
            #     test_pred_y = torch.squeeze(test_pred_y)
            #     tests.append(test_pred_y)
            # test_pred_y = torch.stack(tests, dim=0)
            # for val in test_y:
            #     tests_y.append(val)
            # test_y = torch.stack(tests_y, dim=0)
            f1, acc = scuff_accuracy(test_pred_y, test_y)
            # wandb.log({"test_acc": acc, "test_f1": f1})
            print("epoch: " + str(epoch) + "\nacc: " + str(acc) + "\nf1: " + str(f1) + "\n")

    with torch.no_grad():
        model.eval()
        vals = []
        for t in vals:
            t = t[None, :, :]
            test_pred_y, _ = model(t.to(device))
            test_pred_y = torch.squeeze(test_pred_y)
            vals.append(test_pred_y)
        val_pred_y = torch.stack(vals, dim=0)
        f1, acc = scuff_accuracy(val_pred_y, val_y)
        print("Test metrics:\nacc: " + str(acc) + "\nf1: " + str(f1) + "\n")
    torch.save(model.state_dict(), "mfcc_model/new_mfcc.pt")


if __name__ == "__main__":
    config = {
        "optimizer": "Adam",
        "lr": 0.0001,
        "model": "LSTM_old_small",
        "batch_size": 16,
        "data": "ECF",
        "task": "emotions",
        "mfcc": "transposed",
    }
    # vectors_creation.create_eval_vectors()
    main(config)
    # denoise()
    # create_all_vectors()
    # create_trial_vectors()
