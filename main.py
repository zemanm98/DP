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


def validate_input(args):
    '''
    Method validates the input parameters and exits if the parameters are used incorrectly
    '''
    if args.modality not in ["audio", "text", "multimodal"]:
        print("-modality " + args.modality + " unknown. There are 3 options for modality: audio, text, multimodal\n")
        exit(0)
    if args.dataset not in ["ECF", "IEMOCAP", "RAVDESS", "ECF_FT2D", "ECF_REPETSIM"]:
        print("Unknown dataset. Dataset choices are: ECF or IEMOCAP or RAVDESS. And if any noise reduction method\n"
              "should be used on ECF dataset, its ECF_FT2D and ECF_REPETSIM\n")
        exit(0)
    if args.modality == "audio":
        if args.audio_model is None or args.audio_feature_extraction is None:
            print("When audio modality is chosen, arguments -audio_model and -audio_feature_extraction needs to"
                  "be specified.\n")
            exit(0)
        if args.audio_model not in ["CNN1D", "CNN2D", "MLP"]:
            print("Unknown audio model. Audio model choices are: CNN1D or CNN2D or MLP \n")
            exit(0)
        if args.audio_feature_extraction not in ["collective_features", "mfcc_only"]:
            print("Unknown audio feature extraction method.\nAudio feature extraction method choices are:"
                  "collective_features or mfcc_only\n")
            exit(0)
        return {"audio_model": args.audio_model, "audio_features": args.audio_feature_extraction}
    if args.modality == "text":
        if args.text_model is None or args.text_feature_extraction is None:
            print("When text modality is chosen, arguments -text_model and -text_feature_extraction needs to"
                  "be specified.\n")
            exit(0)
        if args.dataset == "RAVDESS":
            print("RAVDESS does not have text transcriptions, for text emotion recognition choose either ECF or"
                  "IEMOCAP dataset.\n")
            exit(0)
        if args.dataset == "ECF_FT2D" or args.dataset == "ECF_REPETSIM":
            print("The noise reduction method " + args.dataset[4:] + " is used for the audio feature extraction that is"
                                                                     "not used in text modality.\nECF dataset will be used.\n")
        if args.text_model not in ["BERT", "LSTM"]:
            print("Unknown text model. Text model choices are: BERT or LSTM\n")
            exit(0)
        if args.text_feature_extraction not in ["w2v", "bert"]:
            print("Unknown text feature extraction method.\nText feature extraction method choices are: w2v or bert\n")
            exit(0)
        if args.text_model == "BERT" and args.text_feature_extraction == "w2v":
            print("BERT model is not capable of working with w2v feature extraction."
                  "BERT text feature extraction will be used.\n")
        return {"text_model": args.text_model, "text_features": args.text_feature_extraction, "audio_model": None,
                "audio_features": None, "use_audio_model": False}
    if args.modality == "multimodal":
        if args.audio_feature_extraction is None or args.text_model is None or args.text_feature_extraction is None or args.use_audio_model is None:
            print(
                "When multimodal modality is chosen, arguments -text_model, -text_feature_extraction, -use_audio_model and "
                "-audio_feature_extraction needs to be specified.\n")
            exit(0)

        use_audio_model = args.use_audio_model
        if use_audio_model.lower() not in ["true", "false"]:
            print("-use_audio_model argument can be either tru ot false. Nothing else.")
            exit(0)
        if use_audio_model.lower() == "true":
            use_audio_model = True
        else:
            use_audio_model = False

        if use_audio_model and args.audio_model is None:
            print("When -use_audio_model is set to true, -audio_model argument needs to be specified.\n")
            exit(0)
        if args.dataset == "RAVDESS":
            print("RAVDESS does not have text transcriptions, for multimodal emotion recognition choose either ECF,"
                  "ECF_FT2D, ECF_REPETSIM or IEMOCAP dataset.\n")
            exit(0)
        if args.text_model not in ["BERT", "LSTM"]:
            print("Unknown text model. Text model choices are: BERT or LSTM\n")
            exit(0)
        if args.text_feature_extraction not in ["w2v", "bert"]:
            print("Unknown text feature extraction method.\nText feature extraction method choices are: w2v or bert\n")
            exit(0)
        if args.text_model == "BERT" and args.text_feature_extraction == "w2v":
            print("BERT model is not capable of working with w2v feature extraction."
                  "BERT text feature extraction will be used.\n")
        if args.audio_model is not None and args.audio_model not in ["CNN1D", "CNN2D", "MLP"]:
            print("Unknown audio model. Audio model choices are: CNN1D or CNN2D or MLP \n")
            exit(0)
        if args.audio_feature_extraction not in ["collective_features", "mfcc_only"]:
            print("Unknown audio feature extraction method.\nAudio feature extraction method choices are:"
                  "collective_features or mfcc_only\n")
            exit(0)
        return {"audio_model": args.audio_model, "audio_features": args.audio_feature_extraction,
                "text_model": args.text_model, "text_features": args.text_feature_extraction,
                "use_audio_model": use_audio_model}


def text_training(modality, dataset, configuration):
    print("Chosen modality: " + modality + "\n")
    if modality == "audio":
        audio_emotion_learn(configuration["audio_model"], dataset, configuration["audio_features"])
        exit(0)
    if modality == "text":
        if configuration["text_model"] == "LSTM":
            train_LSTM(modality, dataset, configuration)
        else:
            train_bert(modality, dataset, configuration)
    else:
        if configuration["text_model"] == "LSTM":
            train_LSTM(modality, dataset, configuration)
        else:
            train_bert(modality, dataset, configuration)


def train_LSTM(modality, dataset, configuration):
    if modality == "text":
        model = LSTM_text_emotions(dataset, configuration["text_features"], True, False, None)
    else:
        model = LSTM_text_emotions(dataset, configuration["text_features"], False, configuration["use_audio_model"],
                                   configuration["audio_features"])
        if configuration["use_audio_model"]:
            audio_model_name = configuration["audio_model"] + "_" + dataset + "_" + configuration[
                "audio_features"] + ".pt"
            # if the audio feature extraction model does not exist, train it first
            if not os.path.isfile("audio_models/" + audio_model_name) and configuration["use_audio_model"]:
                print("Audio model save point not found. Audio model learning process begun.\n")
                audio_emotion_learn(configuration["audio_model"], dataset, configuration["audio_features"])

    # using the specified text feature extraction method
    if configuration["text_features"] == "w2v":
        word_id_mapping, word_embedding = load_w2v(300, W2V_FILE_PATH, dataset)
        word_embedding = torch.from_numpy(word_embedding)
        train_x, train_y, test_x, test_y, dev_x, dev_y, train_audio, test_audio, dev_audio = load_text_data(
            word_id_mapping, word_embedding,
            30, dataset, configuration["audio_model"], configuration["audio_features"],
            configuration["use_audio_model"])
    else:
        train_x, train_y, test_x, test_y, dev_x, dev_y, train_audio, test_audio, dev_audio = \
            load_text_data_bert(30, dataset, configuration["audio_model"], configuration["audio_features"],
                                configuration["use_audio_model"])

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
    for epoch in range(1, TEXT_AND_MULTIMODAL_EPOCHS_LSTM):
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

    if modality == "text":
        if not os.path.exists("./text_models"):
            os.makedirs("./text_models")
        torch.save(model.state_dict(), "text_models/LSTM_" + dataset + "_" + configuration[
            "text_features"] + ".pt")
    else:
        if not os.path.exists("./multimodal_models"):
            os.makedirs("./multimodal_models")
        if configuration["use_audio_model"]:
            torch.save(model.state_dict(), "multimodal_models/LSTM_" + dataset + "_" + configuration["text_features"] +
                       "_" + configuration["audio_model"] + "_" + configuration["audio_features"] + ".pt")
        else:
            torch.save(model.state_dict(), "multimodal_models/LSTM_" + dataset + "_" + configuration["text_features"] +
                       "_" + configuration["audio_features"] + ".pt")


def train_bert(modality, dataset, configuration):
    if modality == "text":
        custom_model = CustomBert(dataset, True, False, None)
    else:
        custom_model = CustomBert(dataset, False, configuration["use_audio_model"], configuration["audio_features"])
        if configuration["use_audio_model"]:
            audio_model_name = configuration["audio_model"] + "_" + dataset + "_" + configuration[
                "audio_features"] + ".pt"
            # if the audio feature extraction model does not exist, train it first
            if not os.path.isfile("audio_models/" + audio_model_name) and configuration["use_audio_model"]:
                print("Audio model save point not found. Audio model learning process begun.\n")
                audio_emotion_learn(configuration["audio_model"], dataset, configuration["audio_features"])

    # loading the dataset data for the BERT model
    train_inputs, train_attention, train_labels, test_inputs, test_labels, \
    test_attention, dev_inputs, dev_attention, dev_labels, train_audio, test_audio, dev_audio = load_data_for_bert(
        dataset,
        configuration["audio_model"],
        configuration["audio_features"],
        configuration["use_audio_model"])
    train_dataset = TensorDataset(train_inputs, train_attention, train_labels, train_audio)
    test_dataset = TensorDataset(test_inputs, test_attention, test_labels, test_audio)
    dev_dataset = TensorDataset(dev_inputs, dev_attention, dev_labels, dev_audio)

    loader = DataLoader(train_dataset, shuffle=True, batch_size=BERT_BATCH_SIZE)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BERT_BATCH_SIZE)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=BERT_BATCH_SIZE)
    custom_model.to(device)
    optimizer = torch.optim.AdamW(custom_model.parameters(), lr=BERT_LR, eps=1e-8)
    loss_f = torch.nn.MSELoss()
    epochs = TEXT_AND_MULTIMODAL_EPOCHS_BERT
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
                train_test_dataset = TensorDataset(train_test_inputs, train_test_att, train_test_labels,
                                                   train_test_audio)
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

    if modality == "text":
        if not os.path.exists("./text_models"):
            os.makedirs("./text_models")
        torch.save(custom_model.state_dict(), "text_models/BERT_" + dataset + ".pt")
    else:
        if not os.path.exists("./multimodal_models"):
            os.makedirs("./multimodal_models")
        if configuration["use_audio_model"]:
            torch.save(custom_model.state_dict(), "multimodal_models/BERT_" + dataset + "_" + configuration["audio_model"] +
                       "_" + configuration["audio_features"] + ".pt")
        else:
            torch.save(custom_model.state_dict(),
                       "multimodal_models/BERT_" + dataset + "_" + configuration["audio_features"] + ".pt")


if __name__ == "__main__":
    # parsing the input parameters of the script
    parser = argparse.ArgumentParser()
    parser.add_argument('-modality', required=True, type=str)
    parser.add_argument('-text_model', required=False, type=str)
    parser.add_argument('-text_feature_extraction', required=False, type=str)
    parser.add_argument('-audio_model', required=False, type=str)
    parser.add_argument('-audio_feature_extraction', required=False, type=str)
    parser.add_argument('-dataset', required=True, type=str)
    parser.add_argument('-use_audio_model', required=False, type=str)
    args = parser.parse_args()
    # validating input arguments
    conf = validate_input(args)
    text_training(args.modality, args.dataset, conf)
