import argparse

import torch.nn
from torch.utils.data import DataLoader

from config.config import *
from dataset_loading import *
from models import LSTM_text_emotions, CustomBert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils.func import f1_acc
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset
from audio_learning import audio_emotion_learn


def validate_input(text_model, audio_model, text_features, audio_features, dataset):
    '''
    Method validates the input parameters and exits if the parameters are used incorrectly
    '''
    if dataset == "RAVDESS":
        print("RAVDESS does not have transcriptions, so only audio model learning will be deployed.\n")
        return

    if dataset not in ["ECF", "IEMOCAP", "RAVDESS", "ECF_FT2D", "ECF_REPETSIM"]:
        print("Unknown dataset. Dataset choices are: ECF or IEMOCAP or RAVDESS. And if any noise reduction method\n"
              "should be used on ECF dataset, its ECF_FT2D and ECF_REPETSIM\n")
        exit(0)

    if audio_features not in ["collective_features", "mfcc_only"]:
        print(
            "Unknown audio feature extraction method.\nAudio feature extraction method choices are: collective_features or mfcc_only\n")
        exit(0)

    if text_features not in ["w2v", "bert"]:
        print("Unknown text feature extraction method.\nText feature extraction method choices are: w2v or bert\n")
        exit(0)

    if audio_model not in ["CNN1D", "CNN2D", "NN", "LSTM"]:
        print("Unknown audio model. Audio model choices are: CNN1D or CNN2D or NN or LSTM\n")
        exit(0)

    if text_model not in ["BERT", "LSTM"]:
        print("Unknown text model. Text model choices are: BERT or LSTM\n")
        exit(0)

    if text_model == "BERT" and text_features == "w2v":
        print(
            "BERT model is not capable of working with w2v feature extraction. BERT text feature extraction will be used.")
        return


def text_training(text_model, audio_model, text_features, audio_features, dataset, text_only):
    # validating the input parameters
    validate_input(text_model, audio_model, text_features, audio_features, dataset)
    print("Input parameters validated\nMultimodal model: " + text_model + "\nAudio model: " + audio_model +
          "\nText features: " + text_features + "\nAudio features: " + audio_features + "\nDataset: " + dataset + "\n")
    # The RAVDESS dataset does not have a text transcription so only audio learning is employed
    if dataset == "RAVDESS":
        audio_emotion_learn(audio_model, dataset, audio_features)
    else:
        if text_model == "LSTM":
            train_LSTM(audio_model, text_features, audio_features, dataset, text_only)
        else:
            train_bert(audio_model, audio_features, dataset, text_only)


def train_LSTM(audio_model, text_features, audio_features, dataset, text_only):
    model = LSTM_text_emotions(dataset, text_features, text_only)
    audio_model_name = audio_model + "_" + dataset + "_" + audio_features + ".pt"
    # if the audio feature extraction model does not exist, train it first
    if not os.path.isfile("audio_models/" + audio_model_name):
        print("Audio model save point not found. Audio model learning process begun.\n")
        audio_emotion_learn(audio_model, dataset, audio_features)

    # using the specified text feature extraction method
    if text_features == "w2v":
        word_id_mapping, word_embedding = load_w2v(300, W2V_FILE_PATH, dataset)
        word_embedding = torch.from_numpy(word_embedding)
        train_x, train_y, test_x, test_y, dev_x, dev_y, train_audio, test_audio, dev_audio = load_text_data(
            word_id_mapping, word_embedding,
            30, dataset, audio_model, audio_features)
    else:
        train_x, train_y, test_x, test_y, dev_x, dev_y, train_audio, test_audio, dev_audio =\
            load_text_data_bert(30, dataset, audio_model, audio_features)

    # train evaluation step interval for the wandb logging
    if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
        train_eval_step = 35
    else:
        train_eval_step = 18

    loader = DataLoader(list(zip(train_x, train_y, train_audio)), shuffle=True, batch_size=32)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_f = torch.nn.MSELoss()
    tr_test_batch_x = None
    tr_test_batch_y = None
    tr_test_batch_a = None
    train_step_counter = 1
    test_step_counter = 1
    for epoch in range(1, 50):
        model.train()
        counter = 0
        for tr_x_batch, tr_y_batch, tr_batch_audio in loader:
            optimizer.zero_grad()
            tr_pred_y = model(tr_x_batch.to(device), tr_batch_audio.to(device))
            # preparing the train evaluation dataset for train accuracy and train f1 score
            if tr_test_batch_x is None:
                tr_test_batch_a = tr_batch_audio
                tr_test_batch_x = tr_x_batch
                tr_test_batch_y = tr_y_batch
            else:
                tr_test_batch_x = torch.cat((tr_test_batch_x, tr_x_batch), dim=0)
                tr_test_batch_y = torch.cat((tr_test_batch_y, tr_y_batch), dim=0)
                tr_test_batch_a = torch.cat((tr_test_batch_a, tr_batch_audio), dim=0)

            tr_y_batch_f = tr_y_batch.float()
            loss = loss_f(tr_y_batch_f.to(device), tr_pred_y.float())
            loss.backward()
            optimizer.step()
            counter += 1
            if counter % train_eval_step == 0:
                model.eval()
                train_pred_y = model(tr_test_batch_x.to(device), tr_test_batch_a.to(device))
                f1, acc = f1_acc(train_pred_y, tr_test_batch_y, dataset)
                train_step_counter += 1
                print("train acc: " + str(acc) + ";  train f1: " + str(f1) + ";  loss: " + str(loss))
                tr_test_batch_x = None
                tr_test_batch_y = None
                tr_test_batch_a = None
                model.train()

        with torch.no_grad():
            model.eval()
            test_step_counter += 1
            dev_pred_y = model(dev_x.to(device), dev_audio.to(device))
            f1, acc = f1_acc(dev_pred_y, dev_y, dataset)
            print("epoch: " + str(epoch) + "\nacc: " + str(acc) + "\nf1: " + str(f1) + "\n")

    with torch.no_grad():
        model.eval()
        test_pred_y = model(test_x.to(device), test_audio.to(device))
        f1, acc = f1_acc(test_pred_y, test_y, dataset)
        print("\ntest acc: " + str(acc) + "\ntest f1: " + str(f1) + "\n")

    if not os.path.exists("./text_models"):
        os.makedirs("./text_models")
    torch.save(model.state_dict(), "text_models/LSTM_" + dataset + "_" + text_features + "_" + audio_model + "_" + audio_features + ".pt")


def train_bert(audio_model, audio_feature, dataset, text_only):
    audio_model_name = audio_model + "_" + dataset + "_" + audio_feature + ".pt"
    if not os.path.isfile("mfcc_model/" + audio_model_name):
        print("Audio model save point not found. Audio model learning process begun.\n")
        audio_emotion_learn(audio_model, dataset, audio_feature)

    custom_model = CustomBert(dataset, text_only)
    # loading the dataset data for the BERT model
    train_inputs, train_attention, train_labels, test_inputs, test_labels, \
    test_attention, dev_inputs, dev_attention, dev_labels, train_audio, test_audio, dev_audio = load_data_for_bert(dataset,
                                                                                                                   audio_model,
                                                                                                                   audio_feature)
    train_dataset = TensorDataset(train_inputs, train_attention, train_labels, train_audio)
    test_dataset = TensorDataset(test_inputs, test_attention, test_labels, test_audio)
    dev_dataset = TensorDataset(dev_inputs, dev_attention, dev_labels, dev_audio)

    loader = DataLoader(train_dataset, shuffle=True, batch_size=BERT_BATCH_SIZE)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BERT_BATCH_SIZE)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=BERT_BATCH_SIZE)
    custom_model.to(device)
    optimizer = torch.optim.AdamW(custom_model.parameters(), lr=BERT_LR, eps=1e-8)
    loss_f = torch.nn.MSELoss()
    epochs = 3
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    train_test_inputs = None
    train_test_att = None
    train_test_labels = None
    train_test_audio = None
    # train evaluation step interval for the train accuracy and train f1 score
    if dataset == "ECF":
        train_eval_step = 4
        test_eval_step = 10
    else:
        train_eval_step = 2
        test_eval_step = 5
    train_step_counter = 1
    test_step_counter = 1

    for epoch in range(1, epochs):
        custom_model.train()
        counter = 0
        for step, batch in enumerate(loader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_audio = batch[3].to(device)
            optimizer.zero_grad()
            output = custom_model(b_input_ids, b_input_mask, b_audio)
            if train_test_inputs is None:
                train_test_inputs = b_input_ids
                train_test_att = b_input_mask
                train_test_labels = b_labels
                train_test_audio = b_audio
            else:
                train_test_inputs = torch.cat((train_test_inputs, b_input_ids), dim=0)
                train_test_att = torch.cat((train_test_att, b_input_mask), dim=0)
                train_test_labels = torch.cat((train_test_labels, b_labels), dim=0)
                train_test_audio = torch.cat((train_test_audio, b_audio), dim=0)
            loss = loss_f(b_labels.float(), output.float())
            loss.backward()
            optimizer.step()
            scheduler.step()
            counter += 1
            if counter % train_eval_step == 0:
                custom_model.eval()
                train_test_dataset = TensorDataset(train_test_inputs, train_test_att, train_test_labels, train_test_audio)
                train_test_loader = DataLoader(train_test_dataset, shuffle=True, batch_size=BERT_BATCH_SIZE)
                total_eval_accuracy = 0
                total_eval_f1 = 0
                for batch in train_test_loader:
                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)
                    b_labels = batch[2].to(device)
                    b_audio = batch[3].to(device)
                    with torch.no_grad():
                        output = custom_model(b_input_ids, b_input_mask, b_audio)
                    f1, acc = f1_acc(output, b_labels, dataset)
                    total_eval_accuracy += acc
                    total_eval_f1 += f1
                avg_val_accuracy = total_eval_accuracy / len(train_test_loader)
                avg_val_f1 = total_eval_f1 / len(train_test_loader)
                train_step_counter += 1
                print("train acc: " + str(avg_val_accuracy) + " f1: " + str(avg_val_f1) + " loss: " + str(loss))
                train_test_inputs = None
                train_test_att = None
                train_test_labels = None
                train_test_audio = None

            if counter % test_eval_step == 0:
                total_eval_accuracy = 0
                total_eval_f1 = 0
                custom_model.eval()
                for batch in dev_loader:
                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)
                    b_labels = batch[2].to(device)
                    b_audio = batch[3].to(device)
                    with torch.no_grad():
                        output = custom_model(b_input_ids, b_input_mask, b_audio)
                    f1, acc = f1_acc(output, b_labels, dataset)
                    total_eval_accuracy += acc
                    total_eval_f1 += f1
                avg_val_accuracy = total_eval_accuracy / len(dev_loader)
                avg_val_f1 = total_eval_f1 / len(dev_loader)
                test_step_counter += 1
                print("epoch: " + str(epoch) + "\nacc: " + str(avg_val_accuracy) + "\nf1: " + str(avg_val_f1) + "\n")

    total_eval_accuracy = 0
    total_eval_f1 = 0
    custom_model.eval()
    for batch in test_loader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_audio = batch[3].to(device)
        with torch.no_grad():
            output = custom_model(b_input_ids, b_input_mask, b_audio)
        f1, acc = f1_acc(output, b_labels, dataset)
        total_eval_accuracy += acc
        total_eval_f1 += f1
    avg_val_accuracy = total_eval_accuracy / len(test_loader)
    avg_val_f1 = total_eval_f1 / len(test_loader)
    print("test acc: " + str(avg_val_accuracy) + "\n test f1: " + str(avg_val_f1) + "\n")
    torch.save(custom_model.state_dict(), "text_models/BERT_" + dataset + "_" + audio_model + "_" + audio_feature +".pt")


if __name__ == "__main__":
    # parsing the input parameters of the script
    parser = argparse.ArgumentParser()
    parser.add_argument('-text_model', required=True, type=str)
    parser.add_argument('-text_feature_extraction', required=True, type=str)
    parser.add_argument('-audio_model', required=True, type=str)
    parser.add_argument('-audio_feature_extraction', required=True, type=str)
    parser.add_argument('-dataset', required=True, type=str)
    parser.add_argument('-text_only', required=True, type=str)
    args = parser.parse_args()
    text_only = args.text_only
    if text_only.lower() == "true":
        text_only = True
    else:
        text_only = False
    text_training(args.text_model, args.audio_model, args.text_feature_extraction, args.audio_feature_extraction,
                  args.dataset, text_only)
