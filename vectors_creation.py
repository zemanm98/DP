import os
import torch
import numpy as np
import librosa
import json

from models import LSTM_extractor, LSTM_extractor_old, LSTM_extractor_w2v

f = open("data/Subtask_2_2_train.json")
data = json.load(f)

h = open("data/Subtask_2_trial.json")
trial_data = json.load(h)

j = open("data/Subtask_2_test.json")
eval_data = json.load(j)

def create_all_vectors():
    print("Creating mfcc vectors\n")
    output_dile = open("audio_vectors_nfcc2k.tsv", "w", encoding='utf-8')
    model = LSTM_extractor()
    model.load_state_dict(torch.load("mfcc_model/mfcc.pt"))
    model.eval()
    test_files = os.listdir("test")
    test_conversation_ids = []
    for name in test_files:
        id = name[3:].split("utt")[0]
        if id not in test_conversation_ids:
            test_conversation_ids.append(id)

    train_files = os.listdir("train")
    train_conversation_ids = []
    for name in train_files:
        id = name[3:].split("utt")[0]
        if id not in train_conversation_ids:
            train_conversation_ids.append(id)

    val_files = os.listdir("val")
    val_conversation_ids = []
    for name in val_files:
        id = name[3:].split("utt")[0]
        if id not in val_conversation_ids:
            val_conversation_ids.append(id)

    for entry in data:
        print(entry['conversation_ID'])
        if str(entry['conversation_ID']) in train_conversation_ids:
            for conv in entry['conversation']:
                file = "train" + "/dia" + str(entry['conversation_ID']) + "utt" + str(conv['utterance_ID']) + ".wav"
                audio_x, sample_rate = librosa.load(file)
                inputs = np.zeros((1, 50))
                mfcc = np.mean(librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=50).T, axis=0)
                inputs[0] = mfcc
                tensor_inputs = torch.from_numpy(inputs)
                _, hidden_states = model(tensor_inputs.float())
                row = str(entry['conversation_ID']) + "_" + str(conv['utterance_ID'])
                for number in hidden_states[0].detach().numpy():
                    row += "," + str(number)
                output_dile.write(row + "\n")

        elif str(entry['conversation_ID']) in test_conversation_ids:
            for conv in entry['conversation']:
                file = "test" + "/dia" + str(entry['conversation_ID']) + "utt" + str(conv['utterance_ID']) + ".wav"
                audio_x, sample_rate = librosa.load(file)
                inputs = np.zeros((1, 50))
                mfcc = np.mean(librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=50).T, axis=0)
                inputs[0] = mfcc
                tensor_inputs = torch.from_numpy(inputs)
                _, hidden_states = model(tensor_inputs.float())
                row = str(entry['conversation_ID']) + "_" + str(conv['utterance_ID'])
                for number in hidden_states[0].detach().numpy():
                    row += "," + str(number)
                output_dile.write(row + "\n")
        else:
            for conv in entry['conversation']:
                file = "val" + "/dia" + str(entry['conversation_ID']) + "utt" + str(conv['utterance_ID']) + ".wav"
                audio_x, sample_rate = librosa.load(file)
                inputs = np.zeros((1, 50))
                mfcc = np.mean(librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=50).T, axis=0)
                inputs[0] = mfcc
                tensor_inputs = torch.from_numpy(inputs)
                _, hidden_states = model(tensor_inputs.float())
                row = str(entry['conversation_ID']) + "_" + str(conv['utterance_ID'])
                for number in hidden_states[0].detach().numpy():
                    row += "," + str(number)
                output_dile.write(row + "\n")

def create_trial_vectors():
    output_dile = open("audio_vectors_trial_nfcc2k.tsv", "w", encoding='utf-8')
    model = LSTM_extractor()
    model.load_state_dict(torch.load("mfcc_model/mfcc.pt"))
    model.eval()
    for trial in trial_data:
        for conv in trial["conversation"]:
            file = "trial" + "/dia" + str(trial['conversation_ID']) + "utt" + str(conv['utterance_ID']) + ".wav"
            audio_x, sample_rate = librosa.load(file)
            inputs = np.zeros((1, 50))
            mfcc = np.mean(librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=50).T, axis=0)
            inputs[0] = mfcc
            tensor_inputs = torch.from_numpy(inputs)
            _, hidden_states = model(tensor_inputs.float())
            row = str(trial['conversation_ID']) + "_" + str(conv['utterance_ID'])
            for number in hidden_states[0].detach().numpy():
                row += "," + str(number)
            output_dile.write(row + "\n")

def create_eval_vectors():
    output_dile = open("audio_vectors_eval_mfcc2k.tsv", "w", encoding='utf-8')
    model = LSTM_extractor_old()
    model.load_state_dict(torch.load("mfcc_model/mfcc.pt"))
    model.eval()
    for trial in eval_data:
        for conv in trial["conversation"]:
            file = "eval" + "/dia" + str(trial['conversation_ID']) + "utt" + str(conv['utterance_ID']) + ".wav"
            audio_x, sample_rate = librosa.load(file)
            inputs = np.zeros((1, 50))
            mfcc = np.mean(librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=50).T, axis=0)
            inputs[0] = mfcc
            tensor_inputs = torch.from_numpy(inputs)
            _, hidden_states = model(tensor_inputs.float())
            row = str(trial['conversation_ID']) + "_" + str(conv['utterance_ID'])
            for number in hidden_states[0].detach().numpy():
                row += "," + str(number)
            output_dile.write(row + "\n")