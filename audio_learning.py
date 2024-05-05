import torch
from torch.utils.data import DataLoader
from config.config import *
import argparse
from dataset_loading import load_ECF, load_RAVDESS, load_IEMOCAP
from models import LSTM, ConvNet, CNN_small, simple_NN
from utils.func import f1_acc
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model(model_name, feature_method, dataset):
    '''
    Initialize the audio feature extraction model depending on the dataset, feature method and the model name.
    '''
    if model_name == "CNN1D":
        model = ConvNet(feature_method, dataset)
    elif model_name == "CNN2D":
        model = CNN_small(feature_method, dataset)
    elif model_name == "LSTM":
        model = LSTM(feature_method, dataset)
    else:
        model = simple_NN(feature_method, dataset)

    return model


def initialize_train_tensors(model_name, feature_method, dataset):
    '''
    Method is used for train batch append vectors for train accuracy and train f1 score.
    '''
    if feature_method == "collective_features":
        if model_name == "CNN2D":
            train_x_batch = torch.empty((0, 1, 300, 202), dtype=torch.float32)
        elif model_name == "NN":
            train_x_batch = torch.empty((0, 312), dtype=torch.float32)
        else:
            train_x_batch = torch.empty((0, 1, 312), dtype=torch.float32)
    else:
        if model_name == "CNN2D":
            train_x_batch = torch.empty((0, 1, 300, 50), dtype=torch.float32)
        elif model_name == "NN":
            train_x_batch = torch.empty((0, 50), dtype=torch.float32)
        else:
            train_x_batch = torch.empty((0, 1, 50), dtype=torch.float32)

    if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
        train_y_batch = torch.empty((0, 7), dtype=torch.float32)
    elif dataset == "RAVDESS":
        train_y_batch = torch.empty((0, 8), dtype=torch.float32)
    else:
        train_y_batch = torch.empty((0, 5), dtype=torch.float32)

    return train_x_batch, train_y_batch


def audio_emotion_learn(model_name, dataset, feature_extraction):
    print("Starting audio learning:\nModel: " + model_name + "\nAudio features: " + feature_extraction +
          "\nDataset: " + dataset + "\n\n")
    # config for the wandb logging
    config = {}
    if model_name == "CNN1D":
        config["lr"] = CNN1D_LR
        config["batch_size"] = CNN1D_BATCH_SIZE
    elif model_name == "CNN2D":
        config["lr"] = CNN2D_LR
        config["batch_size"] = CNN2D_BATCH_SIZE
    else:
        config["lr"] = NN_LR
        config["batch_size"] = NN_BATCH_SIZE

    model = initialize_model(model_name, feature_extraction, dataset)

    # loading the correct dataset by the given dataset name.
    if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
        train_x, val_x, train_y, val_y, test_x, test_y = load_ECF(feature_extraction, model_name, dataset)
    elif dataset == "RAVDESS":
        train_x, val_x, train_y, val_y, test_x, test_y = load_RAVDESS(feature_extraction, model_name)
    else:
        train_x, val_x, train_y, val_y, test_x, test_y = load_IEMOCAP(feature_extraction, model_name)

    # transforming the dataset data for model input dimensions.
    if model_name == "CNN2D":
        val_x = val_x[:, None, :, :]
        test_x = test_x[:, None, :, :]
    else:
        val_x = val_x[:, None, :]
        test_x = test_x[:, None, :]

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_f = torch.nn.MSELoss()

    # logging interval. Datasets are of different length.
    if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
        train_eval_step = 70
    elif dataset == "RAVDESS":
        train_eval_step = 7
    else:
        train_eval_step = 37

    loader = DataLoader(list(zip(train_x, train_y)), shuffle=True, batch_size=config["batch_size"])
    train_step_counter = 1
    test_step_counter = 1
    for epoch in range(1, 100):
        counter = 0
        model.train()

        train_x_batch, train_y_batch = initialize_train_tensors(model_name, feature_extraction, dataset)
        for tr_x_batch, tr_y_batch in loader:
            optimizer.zero_grad()
            tr_x_batch = tr_x_batch.float()
            # transforming the batch dimensions to model dimensions
            if model_name == "CNN2D":
                tr_x_batch = tr_x_batch[:, None, :, :]
            elif model_name == "CNN1D" or model_name == "LSTM":
                tr_x_batch = tr_x_batch[:, None, :]
            else:
                train_x_batch = train_x_batch

            tr_pred_y, _ = model(tr_x_batch.to(device))
            tr_pred_y = torch.squeeze(tr_pred_y)
            tr_y_batch_f = tr_y_batch.float()
            train_x_batch = torch.cat((train_x_batch, tr_x_batch), dim=0)
            train_y_batch = torch.cat((train_y_batch, tr_y_batch_f), dim=0)
            loss = loss_f(tr_y_batch_f.to(device), tr_pred_y.float())
            counter += 1
            if counter % train_eval_step == 0:
                model.eval()
                eval_pred_y, _ = model(train_x_batch.to(device))
                eval_pred_y = torch.squeeze(eval_pred_y)
                f1, acc = f1_acc(eval_pred_y, train_y_batch, dataset)
                train_step_counter += 1
                print("train acc: " + str(acc) + ";  train f1: " + str(f1) + ";  loss: " + str(loss))
                train_x_batch, train_y_batch = initialize_train_tensors(model_name, feature_extraction, dataset)
                model.train()

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            dev_pred_y, _ = model(val_x.to(device))
            dev_pred_y = torch.squeeze(dev_pred_y)
            f1, acc = f1_acc(dev_pred_y, val_y, dataset)
            test_step_counter += 1
            print("epoch: " + str(epoch) + "\nacc: " + str(acc) + "\nf1: " + str(f1) + "\n")

    with torch.no_grad():
        model.eval()
        test_pred_y, _ = model(test_x.to(device))
        test_pred_y = torch.squeeze(test_pred_y)
        f1, acc = f1_acc(test_pred_y, test_y, dataset)
        print("Test metrics:\nacc: " + str(acc) + "\nf1: " + str(f1) + "\n")

    if not os.path.exists("./audio_models"):
        os.makedirs("./audio_models")
    torch.save(model.state_dict(), "audio_models/" + model_name + "_" + dataset + "_" + feature_extraction + ".pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True, type=str)
    parser.add_argument('-feature_extraction', required=True, type=str)
    parser.add_argument('-dataset', required=True, type=str)
    args = parser.parse_args()

    audio_emotion_learn(args.model, args.dataset, args.feature_extraction)
