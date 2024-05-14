import json
import os
import random

import librosa
import numpy as np
import nussl
import torch
from transformers import BertTokenizer, BertModel

from config.config import *
from models import ConvNet, CNN_small, simple_NN
from utils.feature_extraction import get_mfcconly, get_mfcconly_n_dim, get_collective_features_n_dim, \
    get_collective_features
from utils.func import embedding_lookup

RAVDESS_emotions = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08"
]

IEMOCAP_emotions = [
    "ang",
    "exc",
    "sad",
    "fru",
    "neu",
]

emotion_categories = [
    "surprise",
    "joy",
    "sadness",
    "neutral",
    "disgust",
    "anger",
    "fear",
]


def load_ecf_structure():
    '''
    Loading the ecf dataset into filtered data dictionary for easier further processing
    '''
    f = open(ECF_TEXT_JSON_PATH)
    data = json.load(f)
    filtered_data = {}
    for h in data:
        for g in h["conversation"]:
            filtered_data[str(h["conversation_ID"]) + "_" + str(g["utterance_ID"])] = g["emotion"]

    return data, filtered_data


def load_iemocap_structure():
    '''
    Loading and filtering of the IEMOCAP dataset.
    '''
    iemocap_emotion_exclude = ["hap", "xxx", "oth", "fea", "sur", "dis"]
    text_transcriptions = {}
    transcriptions = os.listdir(IEMOCAP_TRANSCRIPTIONS_FODER)
    for f in transcriptions:
        transcription = open(IEMOCAP_TRANSCRIPTIONS_FODER + "/" + f).readlines()
        for line in transcription:
            inf = line.split(":")
            if inf[0] not in text_transcriptions:
                text_transcriptions[inf[0].split()[0]] = inf[1]

    emotion_files = os.listdir(IEMOCAP_EMOTIONS_FOLDER)
    iemocap_data = []
    for f in emotion_files:
        dialogue = open(IEMOCAP_EMOTIONS_FOLDER + "/" + f).readlines()
        relevant_lines = []
        for line in dialogue:
            if line[0] == "[":
                relevant_lines.append(line)
        for relevant_line in relevant_lines:
            emotion_record = relevant_line.split("\t")
            if emotion_record[2] not in iemocap_emotion_exclude:
                filename = emotion_record[1] + ".wav"
                emotion = emotion_record[2]
                text = text_transcriptions[emotion_record[1]].strip()
                iemocap_data.append({"filename": filename, "emotion": emotion, "text": text})

    return iemocap_data


def load_IEMOCAP(feature_method, model_name):
    '''
    Loading the IEMOCAP dataset into train, test and dev datasets for the audio learning.
    '''
    print("\nLoading IEMOCAP audio dataset.\n")
    iemocap_data = load_iemocap_structure()
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    val_x = []
    val_y = []
    random_indexes = []
    val_indexes = []
    # dividing the IEMOCAP dataset into train, test dev by random indexes from whole dataset
    while len(random_indexes) < 1000:
        index = random.randint(0, 6784)
        if index not in random_indexes:
            random_indexes.append(index)
    while len(val_indexes) < 1000:
        index = random.randint(0, 6784)
        if index not in val_indexes and index not in random_indexes:
            val_indexes.append(index)
    counter = 0
    for entry in iemocap_data:
        # classification class one hot vector
        y = np.zeros(5)
        emotion = entry["emotion"]
        y[IEMOCAP_emotions.index(emotion)] = 1.0
        file_name = entry["filename"]
        # audio signal feature extraction by the given feature extraction method name
        audio_x, sample_rate = librosa.load(IEMOCAP_AUDIO_FOLDER + "/" + file_name, duration=15)
        if feature_method == "collective_features":
            if model_name == "CNN2D":
                features = get_collective_features_n_dim(audio_x, sample_rate, 300)
            else:
                features = get_collective_features(audio_x, sample_rate)
        else:
            if model_name == "CNN2D":
                features = get_mfcconly_n_dim(audio_x, sample_rate, 300)
            else:
                features = get_mfcconly(audio_x, sample_rate)

        # appending data into right dataset split
        if counter in random_indexes:
            test_x.append(features)
            test_y.append(y)
        elif counter in val_indexes:
            val_x.append(features)
            val_y.append(y)
        else:
            train_x.append(features)
            train_y.append(y)
        counter += 1
        if counter % (int(len(iemocap_data) / 10)) == 0:
            print(str((counter/int(len(iemocap_data) / 10)) * 10) + "% |", end=" ")

    # converting list of numpy arrays of tensors into torch tensors.
    train_y = [torch.from_numpy(train_y[i]) for i in range(0, len(train_y))]
    train_y = torch.stack(train_y)
    test_y = [torch.from_numpy(test_y[i]) for i in range(0, len(test_y))]
    test_y = torch.stack(test_y)
    val_y = [torch.from_numpy(val_y[i]) for i in range(0, len(val_y))]
    val_y = torch.stack(val_y)
    train_x, train_y, test_x, test_y, val_x, val_y = map(torch.tensor, [train_x, train_y, test_x, test_y, val_x, val_y])
    train_x = train_x.float()
    train_y = train_y.float()
    test_x = test_x.float()
    test_y = test_y.float()
    val_x = val_x.float()
    val_y = val_y.float()
    return train_x, val_x, train_y, val_y, test_x, test_y


def load_RAVDESS(feature_method, model_name):
    '''
    Loading of the RAVDESS dataset. Returns train, test and dev dataset
    '''
    print("\nLoading RAVDESS audio dataset.\n")
    folders = os.listdir("RAVDESS")
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    val_x = []
    val_y = []
    random_indexes = []
    val_indexes = []
    overall_length = 0
    for f in folders:
        files = os.listdir(RAVDESS_PATH + "/" + f)
        overall_length += len(files)
    # spliting the dataset into train, test and dev by random indexes from the whole dataset.
    while len(random_indexes) < 200:
        index = random.randint(0, 1399)
        if index not in random_indexes:
            random_indexes.append(index)
    while len(val_indexes) < 200:
        index = random.randint(0, 1399)
        if index not in val_indexes and index not in random_indexes:
            val_indexes.append(index)
    counter = 0
    for f in folders:
        files = os.listdir(RAVDESS_PATH + "/" + f)
        for file in files:
            # one hot vector classification class representation
            y = np.zeros(8)
            emotion = file[6:8]
            y[RAVDESS_emotions.index(emotion)] = 1.0
            # audio signal feature extraction by the given feature extraction method name.
            audio_x, sample_rate = librosa.load(RAVDESS_PATH + "/" + f + "/" + file, duration=15)
            if feature_method == "collective_features":
                if model_name == "CNN2D":
                    features = get_collective_features_n_dim(audio_x, sample_rate, 300)
                else:
                    features = get_collective_features(audio_x, sample_rate)
            else:
                if model_name == "CNN2D":
                    features = get_mfcconly_n_dim(audio_x, sample_rate, 300)
                else:
                    features = get_mfcconly(audio_x, sample_rate)
            if counter in random_indexes:
                test_x.append(features)
                test_y.append(y)
            elif counter in val_indexes:
                val_x.append(features)
                val_y.append(y)
            else:
                train_x.append(features)
                train_y.append(y)
            counter += 1
            if counter % (int(overall_length / 10)) == 0:
                print(str((counter / int(overall_length / 10)) * 10) + "% |", end=" ")

    # Converting the list of numpy arrays and list of tensors into torch tensors.
    train_y = [torch.from_numpy(train_y[i]) for i in range(0, len(train_y))]
    train_y = torch.stack(train_y)
    test_y = [torch.from_numpy(test_y[i]) for i in range(0, len(test_y))]
    test_y = torch.stack(test_y)
    val_y = [torch.from_numpy(val_y[i]) for i in range(0, len(val_y))]
    val_y = torch.stack(val_y)
    train_x, train_y, test_x, test_y, val_x, val_y = map(torch.tensor, [train_x, train_y, test_x, test_y, val_x, val_y])
    train_x = train_x.float()
    train_y = train_y.float()
    test_x = test_x.float()
    test_y = test_y.float()
    val_x = val_x.float()
    val_y = val_y.float()
    return train_x, val_x, train_y, val_y, test_x, test_y


def load_ECF(feature_method, model_name, dataset):
    '''
    Loading of the ECF dataset.
    '''
    print("Loading ECF audio dataset.\n")
    # determining if any of the noise reduction methods should be used
    noise_reduction_method = None
    if len(dataset.split("_")) > 1:
        noise_reduction_method = dataset.split("_")[1]
        print("Noise reduction " + noise_reduction_method + " used.\n")
    train_folder = ECF_TRAIN_FOLDER + "/"
    val_folder = ECF_DEV_FOLDER + "/"
    test_folder = ECF_TEST_FOLDER + "/"
    trains = os.listdir(train_folder)
    vals = os.listdir(val_folder)
    tests = os.listdir(test_folder)
    train = []
    val = []
    train_y = []
    val_y = []
    test_x = []
    test_y = []
    ecf_data, filtered_data = load_ecf_structure()
    # appending the data into the right dataset split. The split is given by the dataset from the audio files location.
    overall_length = len(trains) + len(tests) + len(vals)
    counter = 0
    for f in trains:
        key = f[3:].split("utt")[0] + "_" + f[3:].split("utt")[1].split(".")[0]
        # classification class one hot vector representation
        y = np.zeros(7)
        y[emotion_categories.index(filtered_data[key])] = 1.0
        # audio signal feature extraction
        audio_x, sample_rate = librosa.load(train_folder + f, duration=15)

        # noise reduction application
        if noise_reduction_method is not None:
            audio_x = nussl.AudioSignal(audio_data_array=audio_x, sample_rate=sample_rate)
            if noise_reduction_method == "FT2D":
                ft2d = nussl.separation.primitive.FT2D(audio_x, mask_type='binary')
                audio_x = ft2d()
                audio_x = np.squeeze(audio_x[1].audio_data)
            else:
                separator = nussl.separation.primitive.RepetSim(audio_x, mask_type='binary')
                audio_x = separator()
                audio_x = np.squeeze(audio_x[1].audio_data)
        if feature_method == "collective_features":
            if model_name == "CNN2D":
                features = get_collective_features_n_dim(audio_x, sample_rate, 300)
            else:
                features = get_collective_features(audio_x, sample_rate)
        else:
            if model_name == "CNN2D":
                features = get_mfcconly_n_dim(audio_x, sample_rate, 300)
            else:
                features = get_mfcconly(audio_x, sample_rate)
        train_y.append(y)
        train.append(features)
        counter += 1
        if counter % (int(overall_length / 10)) == 0:
            print(str((counter / int((overall_length / 10))) * 10) + "% |", end=" ")
    for f in vals:
        key = f[3:].split("utt")[0] + "_" + f[3:].split("utt")[1].split(".")[0]
        # classification class one hot vector representation
        y = np.zeros(7)
        y[emotion_categories.index(filtered_data[key])] = 1.0
        # audio signal feature extraction
        audio_x, sample_rate = librosa.load(val_folder + f, duration=15)

        # noise reduction application
        if noise_reduction_method is not None:
            audio_x = nussl.AudioSignal(audio_data_array=audio_x, sample_rate=sample_rate)
            if noise_reduction_method == "FT2D":
                ft2d = nussl.separation.primitive.FT2D(audio_x, mask_type='binary')
                audio_x = ft2d()
                audio_x = np.squeeze(audio_x[1].audio_data)
            else:
                separator = nussl.separation.primitive.RepetSim(audio_x, mask_type='binary')
                audio_x = separator()
                audio_x = np.squeeze(audio_x[1].audio_data)
        if feature_method == "collective_features":
            if model_name == "CNN2D":
                features = get_collective_features_n_dim(audio_x, sample_rate, 300)
            else:
                features = get_collective_features(audio_x, sample_rate)
        else:
            if model_name == "CNN2D":
                features = get_mfcconly_n_dim(audio_x, sample_rate, 300)
            else:
                features = get_mfcconly(audio_x, sample_rate)
        val.append(features)
        val_y.append(y)
        counter += 1
        if counter % (int(overall_length / 10)) == 0:
            print(str((counter / int((overall_length / 10))) * 10) + "% |", end=" ")
    for f in tests:
        key = f[3:].split("utt")[0] + "_" + f[3:].split("utt")[1].split(".")[0]
        # classification class one hot vector representation
        y = np.zeros(7)
        y[emotion_categories.index(filtered_data[key])] = 1.0
        # audio signal feature extraction
        audio_x, sample_rate = librosa.load(test_folder + f, duration=15)

        # noise reduction application
        if noise_reduction_method is not None:
            audio_x = nussl.AudioSignal(audio_data_array=audio_x, sample_rate=sample_rate)
            if noise_reduction_method == "FT2D":
                ft2d = nussl.separation.primitive.FT2D(audio_x, mask_type='binary')
                audio_x = ft2d()
                audio_x = np.squeeze(audio_x[1].audio_data)
            else:
                separator = nussl.separation.primitive.RepetSim(audio_x, mask_type='binary')
                audio_x = separator()
                audio_x = np.squeeze(audio_x[1].audio_data)
        if feature_method == "collective_features":
            if model_name == "CNN2D":
                features = get_collective_features_n_dim(audio_x, sample_rate, 300)
            else:
                features = get_collective_features(audio_x, sample_rate)
        else:
            if model_name == "CNN2D":
                features = get_mfcconly_n_dim(audio_x, sample_rate, 300)
            else:
                features = get_mfcconly(audio_x, sample_rate)
        test_x.append(features)
        test_y.append(y)
        counter += 1
        if counter % (int(overall_length / 10)) == 0:
            print(str((counter / int((overall_length / 10))) * 10) + "% |", end=" ")

    # mapping all the data to torch tensors
    train, train_y, val_y, test_y, val, test_x = map(torch.tensor, [train, train_y, val_y, test_y, val, test_x])
    return train.float(), val.float(), train_y.float(), val_y.float(), test_x.float(), test_y.float()


def load_text_data(word_idx, word_embed, max_sen_len, dataset, audio_model, audio_features, use_audio_model):
    '''
    Loading the multimodal data with the Word2vec text extraction method for the LSTM multimodal model.
    '''
    print("Loading dataset " + dataset + " for multimodal LSTM model with w2v embeddings.")
    if len(dataset.split("_")) > 1:
        noise_reduction_method = dataset.split("_")[1]
        print("Noise reduction " + noise_reduction_method + " used.\n")
    train_folder = ECF_TRAIN_FOLDER + "/"
    test_folder = ECF_TEST_FOLDER + "/"
    trains = os.listdir(train_folder)
    tests = os.listdir(test_folder)
    train_x, train_y, test_x, test_y, dev_x, dev_y, train_a, test_a, dev_a = [], [], [], [], [], [], [], [], []
    # initializing the audio feature extraction model by the audio model name given in the input parameters
    if use_audio_model:
        if audio_model == "CNN1D":
            model = ConvNet(audio_features, dataset)
        elif audio_model == "CNN2D":
            model = CNN_small(audio_features, dataset)
        else:
            model = simple_NN(audio_features, dataset)
        saved_model_name = audio_model + "_" + dataset + "_" + audio_features + ".pt"
        model.load_state_dict(torch.load("./audio_models/" + saved_model_name))
        print("Weights for " + audio_model + " loaded.")

    # setting the number of classification classes, out of vocabulary index for the Word2vec
    # and loading the dataset data.
    if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
        n_class = 7
        out_vocab = 6434
        data = []
        ecf_data, filtered_data = load_ecf_structure()
        for conv in ecf_data:
            for sent in conv["conversation"]:
                data.append(
                    {"filename": sent["video_name"][:-3] + "wav", "emotion": sent["emotion"], "text": sent["text"]})
    else:
        n_class = 5
        out_vocab = 6338
        data = load_iemocap_structure()
        random_indexes = []
        val_indexes = []
        # preparing the indexes for the IEMOCAP dataset train/test/dev split
        while len(random_indexes) < 1130:
            index = random.randint(0, 7531)
            if index not in random_indexes:
                random_indexes.append(index)
        while len(val_indexes) < 1130:
            index = random.randint(0, 7531)
            if index not in val_indexes and index not in random_indexes:
                val_indexes.append(index)


    counter = 0
    for conv in data:
        # one hot classification classes vector
        data_x, data_y = np.zeros(max_sen_len), np.zeros(n_class)
        if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
            data_y[emotion_categories.index(conv["emotion"])] = 1
        else:
            data_y[IEMOCAP_emotions.index(conv["emotion"])] = 1

        # transforming the dataset text into embeddings
        for j, word in enumerate(conv["text"].strip().split()):
            word = word.lower()
            if j >= max_sen_len:
                break
            elif word not in word_idx:
                data_x[j] = out_vocab
            else:
                data_x[j] = int(word_idx[word])

        sentence_embeddings = embedding_lookup(word_embed, data_x)

        file_name = conv["filename"]
        if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
            if use_audio_model:
                x_input = get_audio_vector(audio_features, audio_model, file_name, dataset, use_audio_model)
                out1, out2 = model(x_input.float())
                out2 = out2.squeeze()
                audio_vector = out2.detach()
            else:
                audio_vector = get_audio_vector(audio_features, audio_model, file_name, dataset, use_audio_model)
            if file_name in trains:
                train_a.append(audio_vector)
                train_x.append(sentence_embeddings)
                train_y.append(data_y)
            elif file_name in tests:
                test_a.append(audio_vector)
                test_x.append(sentence_embeddings)
                test_y.append(data_y)
            else:
                dev_a.append(audio_vector)
                dev_x.append(sentence_embeddings)
                dev_y.append(data_y)
        else:
            if use_audio_model:
                x_input = get_audio_vector(audio_features, audio_model, file_name, dataset, use_audio_model)
                out1, out2 = model(x_input.float())
                out2 = out2.squeeze()
                audio_vector = out2.detach()
            else:
                audio_vector = get_audio_vector(audio_features, audio_model, file_name, dataset, use_audio_model)
            if counter in random_indexes:
                test_x.append(sentence_embeddings)
                test_y.append(data_y)
                test_a.append(audio_vector)
            elif counter in val_indexes:
                dev_a.append(audio_vector)
                dev_x.append(sentence_embeddings)
                dev_y.append(data_y)
            else:
                train_a.append(audio_vector)
                train_x.append(sentence_embeddings)
                train_y.append(data_y)
        counter += 1
        if counter % (int(len(data) / 10)) == 0:
            print(str((counter / int((len(data) / 10))) * 10) + "% |", end=" ")

    # Converting the list of numpy arrays or tensors into torch tensors
    train_a = torch.stack((train_a))
    test_a = torch.stack((test_a))
    dev_a = torch.stack((dev_a))
    train_y = torch.tensor(train_y)
    train_x = torch.stack((train_x))
    test_x = torch.stack((test_x))
    test_y = torch.tensor(test_y)
    dev_x = torch.stack((dev_x))
    dev_y = torch.tensor(dev_y)

    return train_x, train_y, test_x, test_y, dev_x, dev_y, train_a, test_a, dev_a


def load_text_data_bert(max_sen_len, dataset, audio_model, audio_features, use_audio_model):
    '''
    Loading the multimodal data with the BERT embeddings text extraction method for the LSTM multimodal model.
    '''
    print("Loading dataset " + dataset + " for multimodal LSTM model with BERT embeddings.")
    if len(dataset.split("_")) > 1:
        noise_reduction_method = dataset.split("_")[1]
        print("Noise reduction " + noise_reduction_method + " used.\n")
    # determining the file path for the unchanged or the noise reduced ECF dataset
    train_folder = ECF_TRAIN_FOLDER + "/"
    test_folder = ECF_TEST_FOLDER + "/"

    bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True, )
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    trains = os.listdir(train_folder)
    tests = os.listdir(test_folder)
    train_x, train_y, test_x, test_y, dev_x, dev_y, train_a, test_a, dev_a = [], [], [], [], [], [], [], [], []
    # initializing the audio feature extraction model by the audio model name given in the input parameters
    if use_audio_model:
        if audio_model == "CNN1D":
            model = ConvNet(audio_features, dataset)
        elif audio_model == "CNN2D":
            model = CNN_small(audio_features, dataset)
        else:
            model = simple_NN(audio_features, dataset)
        saved_model_name = audio_model + "_" + dataset + "_" + audio_features + ".pt"
        model.load_state_dict(torch.load("./audio_models/" + saved_model_name))
        print("Weights for " + audio_model + " loaded.")

    # setting the number of classification classes and loading the dataset data.
    if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
        n_class = 7
        data = []
        ecf_data, filtered_data = load_ecf_structure()
        for conv in ecf_data:
            for sent in conv["conversation"]:
                data.append(
                    {"filename": sent["video_name"][:-3] + "wav", "emotion": sent["emotion"], "text": sent["text"]})
    else:
        n_class = 5
        data = load_iemocap_structure()
        random_indexes = []
        val_indexes = []
        # preparing the indexes for the IEMOCAP dataset train/test/dev split
        while len(random_indexes) < 1130:
            index = random.randint(0, 7531)
            if index not in random_indexes:
                random_indexes.append(index)
        while len(val_indexes) < 1130:
            index = random.randint(0, 7531)
            if index not in val_indexes and index not in random_indexes:
                val_indexes.append(index)

    counter = 0
    for conv in data:
        # one hot classification classes vector
        data_x, data_y = np.zeros((max_sen_len, 768)), np.zeros(n_class)
        if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
            data_y[emotion_categories.index(conv["emotion"])] = 1
        else:
            data_y[IEMOCAP_emotions.index(conv["emotion"])] = 1

        # preparing the text feature extraction with the BERT word embeddings.
        line = conv["text"].strip()
        tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(line, bert_tokenizer)
        embedding = get_bert_embeddings(tokens_tensor, segments_tensors, bert_model)
        for j, emb in enumerate(embedding):
            if j >= max_sen_len:
                break
            data_x[j] = np.array(emb)

        file_name = conv["filename"]
        if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
            if use_audio_model:
                x_input = get_audio_vector(audio_features, audio_model, file_name, dataset, use_audio_model)
                out1, out2 = model(x_input.float())
                out2 = out2.squeeze()
                audio_vector = out2.detach()
            else:
                audio_vector = get_audio_vector(audio_features, audio_model, file_name, dataset, use_audio_model)
            if file_name in trains:
                train_a.append(audio_vector)
                train_x.append(data_x)
                train_y.append(data_y)
            elif file_name in tests:
                test_a.append(audio_vector)
                test_x.append(data_x)
                test_y.append(data_y)
            else:
                dev_a.append(audio_vector)
                dev_x.append(data_x)
                dev_y.append(data_y)
        else:
            if use_audio_model:
                x_input = get_audio_vector(audio_features, audio_model, file_name, dataset, use_audio_model)
                out1, out2 = model(x_input.float())
                out2 = out2.squeeze()
                audio_vector = out2.detach()
            else:
                audio_vector = get_audio_vector(audio_features, audio_model, file_name, dataset, use_audio_model)
            if counter in random_indexes:
                test_x.append(data_x)
                test_y.append(data_y)
                test_a.append(audio_vector)
            elif counter in val_indexes:
                dev_a.append(audio_vector)
                dev_x.append(data_x)
                dev_y.append(data_y)
            else:
                train_a.append(audio_vector)
                train_x.append(data_x)
                train_y.append(data_y)
        counter += 1
        if counter % (int(len(data) / 10)) == 0:
            print(str((counter / int((len(data) / 10))) * 10) + "% |", end=" ")

    # Converting the list of numpy arrays or tensors into torch tensors
    train_a = torch.stack((train_a))
    test_a = torch.stack((test_a))
    dev_a = torch.stack((dev_a))
    train_x, train_y, test_x, test_y, dev_x, dev_y = map(torch.tensor,
                                                         [train_x, train_y, test_x, test_y, dev_x, dev_y])

    return train_x, train_y, test_x, test_y, dev_x, dev_y, train_a, test_a, dev_a


def load_data_for_bert(dataset, audio_model, audio_feature, use_audio_model):
    '''
    Method prepares the training data for the pretrained BERT model.
    '''
    print("Loading dataset " + dataset + " for multimodal BERT model.")
    if len(dataset.split("_")) > 1:
        noise_reduction_method = dataset.split("_")[1]
        print("Noise reduction " + noise_reduction_method + " used.\n")
    train_folder = ECF_TRAIN_FOLDER
    test_folder = ECF_TEST_FOLDER
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    trains = os.listdir(train_folder)
    tests = os.listdir(test_folder)
    train_inputs, train_attention, train_labels, test_inputs, test_labels, \
    test_attention, dev_inputs, dev_attention, dev_labels, train_audio, test_audio, dev_audio = [], [], [], [], [], [], \
                                                                                                [], [], [], [], [], []
    # audio feature extraction model initialization
    if use_audio_model:
        if audio_model == "CNN1D":
            model = ConvNet(audio_feature, dataset)
        elif audio_model == "CNN2D":
            model = CNN_small(audio_feature, dataset)
        else:
            model = simple_NN(audio_feature, dataset)
        saved_model_name = audio_model + "_" + dataset + "_" + audio_feature + ".pt"
        model.load_state_dict(torch.load("./audio_models/" + saved_model_name))
        print("Weights for " + audio_model + " loaded.")

    # setting the number of classification classes, maximum sentence lengths and loading the dataset data.
    if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
        n_class = 7
        max_sen_len = 91
        data = []
        ecf_data, filtered_data = load_ecf_structure()
        for conv in ecf_data:
            for sent in conv["conversation"]:
                data.append(
                    {"filename": sent["video_name"][:-3] + "wav", "emotion": sent["emotion"], "text": sent["text"]})
    else:
        n_class = 5
        max_sen_len = 126
        data = load_iemocap_structure()
        random_indexes = []
        val_indexes = []
        # preparing the indexes for the IEMOCAP dataset train/test/dev split
        while len(random_indexes) < 1130:
            index = random.randint(0, 7531)
            if index not in random_indexes:
                random_indexes.append(index)
        while len(val_indexes) < 1130:
            index = random.randint(0, 7531)
            if index not in val_indexes and index not in random_indexes:
                val_indexes.append(index)

    counter = 0
    max_length = 0
    for conv in data:
        data_y = np.zeros(n_class)
        line = conv["text"].strip()
        if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
            data_y[emotion_categories.index(conv["emotion"])] = 1
        else:
            data_y[IEMOCAP_emotions.index(conv["emotion"])] = 1
        encoded_dict = bert_tokenizer.encode_plus(
            line,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_sen_len,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        if encoded_dict['input_ids'].size()[1] > max_length:
            max_length = encoded_dict['input_ids'].size()[1]
        file_name = conv["filename"]
        if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
            if use_audio_model:
                x_input = get_audio_vector(audio_feature, audio_model, file_name, dataset, use_audio_model)
                out1, out2 = model(x_input.float())
                out2 = out2.squeeze()
                audio_vector = out2.detach()
            else:
                audio_vector = get_audio_vector(audio_feature, audio_model, file_name, dataset, use_audio_model)
            if file_name in trains:
                train_audio.append(audio_vector)
                train_inputs.append(encoded_dict['input_ids'])
                train_attention.append(encoded_dict['attention_mask'])
                train_labels.append(data_y)
            elif file_name in tests:
                test_audio.append(audio_vector)
                test_inputs.append(encoded_dict['input_ids'])
                test_attention.append(encoded_dict['attention_mask'])
                test_labels.append(data_y)
            else:
                dev_audio.append(audio_vector)
                dev_inputs.append(encoded_dict['input_ids'])
                dev_attention.append(encoded_dict['attention_mask'])
                dev_labels.append(data_y)
        else:
            if use_audio_model:
                x_input = get_audio_vector(audio_feature, audio_model, file_name, dataset, use_audio_model)
                out1, out2 = model(x_input.float())
                out2 = out2.squeeze()
                audio_vector = out2.detach()
            else:
                audio_vector = get_audio_vector(audio_feature, audio_model, file_name, dataset, use_audio_model)
            if counter in random_indexes:
                test_audio.append(audio_vector)
                test_inputs.append(encoded_dict['input_ids'])
                test_attention.append(encoded_dict['attention_mask'])
                test_labels.append(data_y)
            elif counter in val_indexes:
                dev_audio.append(audio_vector)
                dev_inputs.append(encoded_dict['input_ids'])
                dev_attention.append(encoded_dict['attention_mask'])
                dev_labels.append(data_y)
            else:
                train_audio.append(audio_vector)
                train_inputs.append(encoded_dict['input_ids'])
                train_attention.append(encoded_dict['attention_mask'])
                train_labels.append(data_y)
        counter += 1
        if counter % (int(len(data) / 10)) == 0:
            print(str((counter / int((len(data) / 10))) * 10) + "% |", end=" ")

    # Converting the list of numpy arrays or tensors into torch tensors
    train_inputs = torch.cat(train_inputs, dim=0)
    train_attention = torch.cat(train_attention, dim=0)
    test_inputs = torch.cat(test_inputs, dim=0)
    test_attention = torch.cat(test_attention, dim=0)
    dev_inputs = torch.cat(dev_inputs, dim=0)
    dev_attention = torch.cat(dev_attention, dim=0)
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)
    dev_labels = torch.tensor(dev_labels)
    train_audio = torch.stack((train_audio))
    test_audio = torch.stack((test_audio))
    dev_audio = torch.stack((dev_audio))
    return train_inputs, train_attention, train_labels, test_inputs, test_labels, \
           test_attention, dev_inputs, dev_attention, dev_labels, train_audio, test_audio, dev_audio


def bert_text_preparation(text, tokenizer):
    '''
    Bert text tokenization.
    '''
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return tokenized_text, tokens_tensor, segments_tensors


def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    '''
    retrieving the bert word embeddings from thew last hidden state of the pretrained BERT model.
    '''
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2][1:]

    token_embeddings = hidden_states[-1]
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]
    return list_token_embeddings


def load_w2v(embedding_dim, embedding_path, dataset):
    print('\nloading embedding vectors\n')
    words = []
    # loading words of specified dataset
    if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
        data, filtered_data = load_ecf_structure()
        for conv in data:
            for sentence in conv["conversation"]:
                words.extend(sentence["text"].lower().strip().split())
    else:
        iemocap_data = load_iemocap_structure()
        for entry in iemocap_data:
            words.extend(entry["text"].lower().strip().split())

    # unique words
    words = set(words)
    # word ids
    word_idx = dict((c, k + 1) for k, c in enumerate(words))

    # getting the embeddings for the words
    w2v = {}
    inputFile1 = open(embedding_path, 'r')
    inputFile1.readline()
    for line in inputFile1.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))
    embedding.extend([list(np.random.rand(embedding_dim) / 5. - 0.1)])
    embedding = np.array(embedding)

    print("embedding.shape: {}".format(embedding.shape))
    print("load embedding done!\n")
    return word_idx, embedding


def get_audio_vector(audio_feature, model_name, filename, dataset, use_audio_model):
    '''
    Creating the audio feature extraction vector depending on the given audio feature extraction method name,
    audio model and dataset name.
    '''
    # determining if any of the noise reduction methods should be used
    noise_reduction_method = None
    if len(dataset.split("_")) > 1:
        noise_reduction_method = dataset.split("_")[1]
    train_folder = ECF_TRAIN_FOLDER + "/"
    val_folder = ECF_DEV_FOLDER + "/"
    test_folder = ECF_TEST_FOLDER + "/"
    trains = os.listdir(train_folder)
    tests = os.listdir(test_folder)
    if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
        if filename in trains:
            audio_x, sample_rate = librosa.load(train_folder + filename, duration=15)

            # noise reduction application
            if noise_reduction_method is not None:
                audio_x = nussl.AudioSignal(audio_data_array=audio_x, sample_rate=sample_rate)
                if noise_reduction_method == "FT2D":
                    ft2d = nussl.separation.primitive.FT2D(audio_x, mask_type='binary')
                    audio_x = ft2d()
                    audio_x = np.squeeze(audio_x[1].audio_data)
                else:
                    separator = nussl.separation.primitive.RepetSim(audio_x, mask_type='binary')
                    audio_x = separator()
                    audio_x = np.squeeze(audio_x[1].audio_data)
        elif filename in tests:
            audio_x, sample_rate = librosa.load(test_folder + filename, duration=15)

            # noise reduction application
            if noise_reduction_method is not None:
                audio_x = nussl.AudioSignal(audio_data_array=audio_x, sample_rate=sample_rate)
                if noise_reduction_method == "FT2D":
                    ft2d = nussl.separation.primitive.FT2D(audio_x, mask_type='binary')
                    audio_x = ft2d()
                    audio_x = np.squeeze(audio_x[1].audio_data)
                else:
                    separator = nussl.separation.primitive.RepetSim(audio_x, mask_type='binary')
                    audio_x = separator()
                    audio_x = np.squeeze(audio_x[1].audio_data)
        else:
            audio_x, sample_rate = librosa.load(val_folder + filename, duration=15)

            # noise reduction application
            if noise_reduction_method is not None:
                audio_x = nussl.AudioSignal(audio_data_array=audio_x, sample_rate=sample_rate)
                if noise_reduction_method == "FT2D":
                    ft2d = nussl.separation.primitive.FT2D(audio_x, mask_type='binary')
                    audio_x = ft2d()
                    audio_x = np.squeeze(audio_x[1].audio_data)
                else:
                    separator = nussl.separation.primitive.RepetSim(audio_x, mask_type='binary')
                    audio_x = separator()
                    audio_x = np.squeeze(audio_x[1].audio_data)

        if use_audio_model:
            if audio_feature == "collective_features":
                if model_name == "CNN2D":
                    audio_data = get_collective_features_n_dim(audio_x, sample_rate, 300)
                else:
                    audio_data = get_collective_features(audio_x, sample_rate)
            else:
                if model_name == "CNN2D":
                    audio_data = get_mfcconly_n_dim(audio_x, sample_rate, 300)
                else:
                    audio_data = get_mfcconly(audio_x, sample_rate)
            x_input = torch.from_numpy(audio_data)
            # reshaping the dimensions for the audio model
            if model_name == "CNN2D":
                x_input = x_input[None, None, :, :]
            elif model_name == "CNN1D":
                x_input = x_input[None, None, :]
            else:
                x_input = x_input[None, :]
        else:
            if audio_feature == "collective_features":
                audio_data = get_collective_features(audio_x, sample_rate)
            else:
                audio_data = get_mfcconly(audio_x, sample_rate)
            x_input = torch.from_numpy(audio_data)
    else:
        audio_x, sample_rate = librosa.load(IEMOCAP_AUDIO_FOLDER + "/" + filename, duration=15)
        if use_audio_model:
            if audio_feature == "collective_features":
                if model_name == "CNN2D":
                    audio_data = get_collective_features_n_dim(audio_x, sample_rate, 300)
                else:
                    audio_data = get_collective_features(audio_x, sample_rate)
            else:
                if model_name == "CNN2D":
                    audio_data = get_mfcconly_n_dim(audio_x, sample_rate, 300)
                else:
                    audio_data = get_mfcconly(audio_x, sample_rate)
            x_input = torch.from_numpy(audio_data)
            # reshaping the dimensions for the audio model
            if model_name == "CNN2D":
                x_input = x_input[None, None, :, :]
            elif model_name == "CNN1D":
                x_input = x_input[None, None, :]
            else:
                x_input = x_input[None, :]
        else:
            if audio_feature == "collective_features":
                audio_data = get_collective_features(audio_x, sample_rate)
            else:
                audio_data = get_mfcconly(audio_x, sample_rate)
            x_input = torch.from_numpy(audio_data)

    return x_input
