import os
import torch
import numpy as np
import vectors_creation
from models import LSTM_extractor, LSTM_extractor_old, ConvNet, CNN_small, simple_NN
from dataset_loading import load_dataset, load_old_dataset, load_RAVDESS, load_IEMOCAP
import soundfile as sf
import norbert
import nussl
import torcheval.metrics.functional
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy, multilabel_accuracy
from torchmetrics import Accuracy, F1Score
from torch.utils.data import DataLoader
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
    f1 = multiclass_f1_score(pred_indices, true_indices, num_classes=7, average="macro")
    acc = multiclass_accuracy(pred_indices, true_indices, num_classes=7, average="macro")
    accuracy = Accuracy(task="multiclass", num_classes=7)
    acc2 = accuracy(pred_indices, true_indices)
    f1macro = F1Score(task="multiclass", num_classes=7, average="macro")
    f1scuff = f1macro(pred_indices, true_indices)
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


def main(config):
    model = LSTM_extractor_old()
    # model = ConvNet()
    # model = CNN_small()
    train_x, val_x, train_y, val_y, train_speakers, val_speakers, test_x, test_y, test_speakers = load_old_dataset()
    val_x = val_x[:, None, :]
    # val_x = val_x[:, None, :, :]
    # test_x = test_x[:, None, :, :] # CNN2d
    test_x = test_x[:, None, :]  # LSTM and CNN1d
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_f = torch.nn.MSELoss()
    # wandb.init(project="DP", entity="zeman339", name="CNN2D_ECF_mfcc_only", config=config)
    # wandb.define_metric("Steps")
    # wandb.define_metric("*", step_metric="Steps")
    loader = DataLoader(list(zip(train_x, train_y)), shuffle=True, batch_size=config["batch_size"])
    train_step_counter = 1
    test_step_counter = 1
    for epoch in range(1, 100):
        counter = 0
        model.train()
        # train_x_batch = torch.empty((0, 1, 300, 50), dtype=torch.float32)
        train_x_batch = torch.empty((0, 1, 312), dtype=torch.float32)
        train_y_batch = torch.empty((0, 7), dtype=torch.float32)
        # for train, _ in get_batch_data_pair(train_x, train_y, config["batch_size"]):
        for tr_x_batch, tr_y_batch in loader:
            # tr_x_batch, tr_y_batch = train
            optimizer.zero_grad()
            tr_x_batch = tr_x_batch.float()
            tr_x_batch = tr_x_batch[:, None, :]
            # tr_x_batch = tr_x_batch[:, None, :, :]
            tr_pred_y, _ = model(tr_x_batch.to(device))
            tr_pred_y = torch.squeeze(tr_pred_y)
            tr_y_batch_f = tr_y_batch.float()
            train_x_batch = torch.cat((train_x_batch, tr_x_batch), dim=0)
            train_y_batch = torch.cat((train_y_batch, tr_y_batch_f), dim=0)
            loss = loss_f(tr_y_batch_f.to(device), tr_pred_y.float())
            counter += 1
            if counter % 70 == 0:
                model.eval()
                # vals = []
                eval_pred_y, _ = model(train_x_batch.to(device))
                eval_pred_y = torch.squeeze(eval_pred_y)
                # for value in val_x:
                #     value = value[None, :, :]
                #     val_pred_y, _ = model(value.to(device))
                #     val_pred_y = torch.squeeze(val_pred_y)
                #     vals.append(val_pred_y)
                # val_pred_y = torch.stack(vals, dim=0)
                # f1, acc = scuff_accuracy(val_pred_y, val_y)
                # # wandb.log({"test_acc": acc, "test_f1": f1})
                # print("epoch: " + str(epoch) + "\nacc: " + str(acc) + "\nf1: " + str(f1) + "\n")
                f1, acc = scuff_accuracy(eval_pred_y, train_y_batch)
                # wandb.log({"train_acc": acc, "train_f1": f1, "Steps": train_step_counter})
                train_step_counter += 1
                print("train acc: " + str(acc) + ";  train f1: " + str(f1) + ";  loss: " + str(loss))
                # train_x_batch = torch.empty((0, 1, 300, 50), dtype=torch.float32)
                train_x_batch = torch.empty((0, 1, 312), dtype=torch.float32)
                train_y_batch = torch.empty((0, 7), dtype=torch.float32)
                model.train()

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            tests = []
            tests_y = []
            dev_pred_y, _ = model(val_x.to(device))
            dev_pred_y = torch.squeeze(dev_pred_y)
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
            f1, acc = scuff_accuracy(dev_pred_y, val_y)
            # wandb.log({"val_acc": acc, "val_f1": f1, "Steps": test_step_counter})
            test_step_counter += 1
            print("epoch: " + str(epoch) + "\nacc: " + str(acc) + "\nf1: " + str(f1) + "\n")

    with torch.no_grad():
        model.eval()
        # vals = []
        # for t in vals:
        #     t = t[None, :, :]
        #     test_pred_y, _ = model(t.to(device))
        #     test_pred_y = torch.squeeze(test_pred_y)
        #     vals.append(test_pred_y)
        test_pred_y, _ = model(test_x.to(device))
        test_pred_y = torch.squeeze(test_pred_y)
        # val_pred_y = torch.stack(vals, dim=0)
        f1, acc = scuff_accuracy(test_pred_y, test_y)
        # wandb.log({"test_acc": acc, "test_f1": f1, "Steps": 1})
        print("Test metrics:\nacc: " + str(acc) + "\nf1: " + str(f1) + "\n")
    torch.save(model.state_dict(), "mfcc_model/LSTM_ecf_collective_features.pt")


if __name__ == "__main__":
    config = {
        "optimizer": "Adam",
        "lr": 0.0001,
        "model": "LSTM",
        "batch_size": 32,
        "data": "ECF",
        "task": "emotions",
        "mfcc": "collective_features",
    }
    # vectors_creation.create_eval_vectors()
    main(config)
    # denoise()
    # create_all_vectors()
    # create_trial_vectors()
