import torch
import os
import numpy as np
import random
import librosa
import json
from utils.feature_extraction import get_mfcconly, get_mfcconly_n_dim, get_collective_features_n_dim, \
    get_collective_features
from utils.func import embedding_lookup
import torchaudio
from transformers import BertTokenizer, BertModel
from transformers import AutoFeatureExtractor
from models import ConvNet, LSTM, CNN_small, simple_NN

f = open("data/Subtask_2_2_train.json")
data_old = json.load(f)
filtered_data = {}
speakers_data = {}
# speakers_occurences = {}
for h in data_old:
    for g in h["conversation"]:
        filtered_data[str(h["conversation_ID"]) + "_" + str(g["utterance_ID"])] = g["emotion"]
        speakers_data[str(h["conversation_ID"]) + "_" + str(g["utterance_ID"])] = g["speaker"]
        # if g["speaker"] not in speakers_occurences:
        #     speakers_occurences[g["speaker"]] = 0
        # speakers_occurences[g["speaker"]] += 1


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

speakers = [
    "Joey",
    "Ross",
    "Rachel",
    "Phoebe",
    "Monica",
    "Chandler"
]


def load_ecf_structure():
    f = open("data/Subtask_2_2_train.json")
    data = json.load(f)
    filtered_data = {}
    for h in data:
        for g in h["conversation"]:
            filtered_data[str(h["conversation_ID"]) + "_" + str(g["utterance_ID"])] = g["emotion"]

    return data, filtered_data

def load_iemocap_structure():
    iemocap_emotion_exclude = ["hap", "xxx", "oth", "fea", "sur", "dis"]
    text_transcriptions = {}
    transcriptions = os.listdir("IEMOCAP/transcription")
    for f in transcriptions:
        transcription = open("IEMOCAP/transcription/"+f).readlines()
        for line in transcription:
            inf = line.split(":")
            if inf[0] not in text_transcriptions:
                text_transcriptions[inf[0].split()[0]] = inf[1]

    emotion_files = os.listdir("IEMOCAP/emotions")
    iemocap_data = []
    for f in emotion_files:
        dialogue = open("IEMOCAP/emotions/"+f).readlines()
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
    iemocap_data = load_iemocap_structure()
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    val_x = []
    val_y = []
    random_indexes = []
    val_indexes = []
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
        y = np.zeros(5)
        emotion = entry["emotion"]
        y[IEMOCAP_emotions.index(emotion)] = 1.0
        file_name = entry["filename"]
        audio_x, sample_rate = librosa.load("IEMOCAP/audio/" + file_name, duration=15)
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
    folders = os.listdir("RAVDESS")
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    val_x = []
    val_y = []
    random_indexes = []
    val_indexes = []
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
        files = os.listdir("RAVDESS/" + f)
        for file in files:
            y = np.zeros(8)
            emotion = file[6:8]
            y[RAVDESS_emotions.index(emotion)] = 1.0
            audio_x, sample_rate = librosa.load("RAVDESS/" + f + "/" + file, duration=15)
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


def load_ECF(feature_method, model_name):
    train_folder = "train"
    val_folder = "val"
    test_folder = "test"
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
    for f in trains:
        key = f[3:].split("utt")[0] + "_" + f[3:].split("utt")[1].split(".")[0]
        y = np.zeros(7)
        y[emotion_categories.index(filtered_data[key])] = 1.0
        audio_x, sample_rate = librosa.load("train/" + f, duration=15)
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
    for f in vals:
        key = f[3:].split("utt")[0] + "_" + f[3:].split("utt")[1].split(".")[0]
        y = np.zeros(7)
        y[emotion_categories.index(filtered_data[key])] = 1.0
        audio_x, sample_rate = librosa.load("val/" + f, duration=15)
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
    for f in tests:
        key = f[3:].split("utt")[0] + "_" + f[3:].split("utt")[1].split(".")[0]
        y = np.zeros(7)
        y[emotion_categories.index(filtered_data[key])] = 1.0
        audio_x, sample_rate = librosa.load("test/" + f, duration=15)
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

    train, train_y, val_y, test_y, val, test_x = map(torch.tensor, [train, train_y, val_y, test_y, val, test_x])
    return train.float(), val.float(), train_y.float(), val_y.float(), test_x.float(), test_y.float()


def load_text_data(word_idx, word_embed, max_sen_len, dataset, audio_model, audio_features):
    train_folder = "train"
    test_folder = "test"
    trains = os.listdir(train_folder)
    tests = os.listdir(test_folder)
    train_x, train_y, test_x, test_y, dev_x, dev_y, train_a, test_a, dev_a = [], [], [], [], [], [], [], [], []
    if audio_model == "LSTM":
        model = LSTM(audio_features, dataset)
    elif audio_model == "CNN1D":
        model = ConvNet(audio_features, dataset)
    elif audio_model == "CNN2D":
        model = CNN_small(audio_features, dataset)
    else:
        model = simple_NN(audio_features, dataset)

    if dataset == "ECF":
        n_class = 7
        out_vocab = 6434
        data = []
        ecf_data, filtered_data = load_ecf_structure()
        for conv in ecf_data:
            for sent in conv["conversation"]:
                data.append({"filename": sent["video_name"][:-3] + "wav", "emotion": sent["emotion"], "text": sent["text"]})
    else:
        n_class = 5
        out_vocab = 6338
        data = load_iemocap_structure()
        random_indexes = []
        val_indexes = []
        while len(random_indexes) < 1130:
            index = random.randint(0, 7531)
            if index not in random_indexes:
                random_indexes.append(index)
        while len(val_indexes) < 1130:
            index = random.randint(0, 7531)
            if index not in val_indexes and index not in random_indexes:
                val_indexes.append(index)

    saved_model_name = audio_model + "_" + dataset + "_" + audio_features + ".pt"
    model.load_state_dict(torch.load("./mfcc_model/" + saved_model_name))
    counter = 0
    for conv in data:
        data_x, data_y = np.zeros(max_sen_len), np.zeros(n_class)
        if dataset == "ECF":
            data_y[emotion_categories.index(conv["emotion"])] = 1
        else:
            data_y[IEMOCAP_emotions.index(conv["emotion"])] = 1

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
        if dataset == "ECF":
            x_input = get_audio_vector(audio_features, audio_model, file_name, dataset)
            out1, out2 = model(x_input.float())
            out2 = out2.squeeze()
            if file_name in trains:
                train_a.append(out2.detach().numpy())
                train_x.append(sentence_embeddings)
                train_y.append(data_y)
            elif file_name in tests:
                test_a.append(out2.detach().numpy())
                test_x.append(sentence_embeddings)
                test_y.append(data_y)
            else:
                dev_a.append(out2.detach().numpy())
                dev_x.append(sentence_embeddings)
                dev_y.append(data_y)
        else:
            x_input = get_audio_vector(audio_features, audio_model, file_name, dataset)
            out1, out2 = model(x_input.float())
            out2 = out2.squeeze()
            if counter in random_indexes:
                test_x.append(sentence_embeddings)
                test_y.append(data_y)
                test_a.append(out2.detach().numpy())
            elif counter in val_indexes:
                dev_a.append(out2.detach().numpy())
                dev_x.append(sentence_embeddings)
                dev_y.append(data_y)
            else:
                train_a.append(out2.detach().numpy())
                train_x.append(sentence_embeddings)
                train_y.append(data_y)
        counter += 1

    train_a = torch.tensor(train_a)
    test_a = torch.tensor(test_a)
    dev_a = torch.tensor(dev_a)
    train_y = torch.tensor(train_y)
    train_x = torch.stack((train_x))
    test_x = torch.stack((test_x))
    test_y = torch.tensor(test_y)
    dev_x = torch.stack((dev_x))
    dev_y = torch.tensor(dev_y)

    return train_x, train_y, test_x, test_y, dev_x, dev_y, train_a, test_a, dev_a


def load_text_data_bert(max_sen_len, dataset, audio_model, audio_features):
    train_folder = "train"
    test_folder = "test"
    bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True, )
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    trains = os.listdir(train_folder)
    tests = os.listdir(test_folder)
    train_x, train_y, test_x, test_y, dev_x, dev_y, train_a, test_a, dev_a = [], [], [], [], [], [], [], [], []

    if audio_model == "LSTM":
        model = LSTM(audio_features, dataset)
    elif audio_model == "CNN1D":
        model = ConvNet(audio_features, dataset)
    elif audio_model == "CNN2D":
        model = CNN_small(audio_features, dataset)
    else:
        model = simple_NN(audio_features, dataset)

    if dataset == "ECF":
        n_class = 7
        data = []
        ecf_data, filtered_data = load_ecf_structure()
        for conv in ecf_data:
            for sent in conv["conversation"]:
                data.append({"filename": sent["video_name"][:-3] + "wav", "emotion": sent["emotion"], "text": sent["text"]})
    else:
        n_class = 5
        data = load_iemocap_structure()
        random_indexes = []
        val_indexes = []
        while len(random_indexes) < 1130:
            index = random.randint(0, 7531)
            if index not in random_indexes:
                random_indexes.append(index)
        while len(val_indexes) < 1130:
            index = random.randint(0, 7531)
            if index not in val_indexes and index not in random_indexes:
                val_indexes.append(index)

    saved_model_name = audio_model + "_" + dataset + "_" + audio_features + ".pt"
    model.load_state_dict(torch.load("./mfcc_model/" + saved_model_name))
    counter = 0
    for conv in data:
        data_x, data_y = np.zeros((max_sen_len, 768)), np.zeros(n_class)
        if dataset == "ECF":
            data_y[emotion_categories.index(conv["emotion"])] = 1
        else:
            data_y[IEMOCAP_emotions.index(conv["emotion"])] = 1
        line = conv["text"].strip()
        tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(line, bert_tokenizer)
        embedding = get_bert_embeddings(tokens_tensor, segments_tensors, bert_model)
        for j, emb in enumerate(embedding):
            if j >= max_sen_len:
                break
            data_x[j] = np.array(emb)

        file_name = conv["filename"]
        if dataset == "ECF":
            x_input = get_audio_vector(audio_features, audio_model, file_name, dataset)
            out1, out2 = model(x_input.float())
            out2 = out2.squeeze()
            if file_name in trains:
                train_a.append(out2.detach().numpy())
                train_x.append(data_x)
                train_y.append(data_y)
            elif file_name in tests:
                test_a.append(out2.detach().numpy())
                test_x.append(data_x)
                test_y.append(data_y)
            else:
                dev_a.append(out2.detach().numpy())
                dev_x.append(data_x)
                dev_y.append(data_y)
        else:
            x_input = get_audio_vector(audio_features, audio_model, file_name, dataset)
            out1, out2 = model(x_input.float())
            out2 = out2.squeeze()
            if counter in random_indexes:
                test_x.append(data_x)
                test_y.append(data_y)
                test_a.append(out2.detach().numpy())
            elif counter in val_indexes:
                dev_a.append(out2.detach().numpy())
                dev_x.append(data_x)
                dev_y.append(data_y)
            else:
                train_a.append(out2.detach().numpy())
                train_x.append(data_x)
                train_y.append(data_y)
        counter += 1

    train_a = torch.tensor(train_a)
    test_a = torch.tensor(test_a)
    dev_a = torch.tensor(dev_a)
    train_x, train_y, test_x, test_y, dev_x, dev_y = map(torch.tensor,
                                                         [train_x, train_y, test_x, test_y, dev_x, dev_y])

    return train_x, train_y, test_x, test_y, dev_x, dev_y, train_a, test_a, dev_a


def load_data_for_bert(dataset, audio_model, audio_feature):
    train_folder = "train"
    test_folder = "test"
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    trains = os.listdir(train_folder)
    tests = os.listdir(test_folder)
    train_inputs, train_attention, train_labels, test_inputs, test_labels, \
    test_attention, dev_inputs, dev_attention, dev_labels, train_audio, test_audio, dev_audio = [], [], [], [], [], [], \
                                                                                                [], [], [], [], [], []
    if audio_model == "LSTM":
        model = LSTM(audio_feature, dataset)
    elif audio_model == "CNN1D":
        model = ConvNet(audio_feature, dataset)
    elif audio_model == "CNN2D":
        model = CNN_small(audio_feature, dataset)
    else:
        model = simple_NN(audio_feature, dataset)

    if dataset == "ECF":
        n_class = 7
        max_sen_len = 91
        data = []
        ecf_data, filtered_data = load_ecf_structure()
        for conv in ecf_data:
            for sent in conv["conversation"]:
                data.append({"filename": sent["video_name"][:-3] + "wav", "emotion": sent["emotion"], "text": sent["text"]})
    else:
        n_class = 5
        max_sen_len = 126
        data = load_iemocap_structure()
        random_indexes = []
        val_indexes = []
        while len(random_indexes) < 1130:
            index = random.randint(0, 7531)
            if index not in random_indexes:
                random_indexes.append(index)
        while len(val_indexes) < 1130:
            index = random.randint(0, 7531)
            if index not in val_indexes and index not in random_indexes:
                val_indexes.append(index)

    saved_model_name = audio_model + "_" + dataset + "_" + audio_feature + ".pt"
    model.load_state_dict(torch.load("./mfcc_model/" + saved_model_name))
    counter = 0
    max_length = 0
    for conv in data:
        data_y = np.zeros(n_class)
        line = conv["text"].strip()
        if dataset == "ECF":
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
        if dataset == "ECF":
            x_input = get_audio_vector(audio_feature, audio_model, file_name, dataset)
            out1, out2 = model(x_input.float())
            out2 = out2.squeeze()
            if file_name in trains:
                train_audio.append(out2.detach().numpy())
                train_inputs.append(encoded_dict['input_ids'])
                train_attention.append(encoded_dict['attention_mask'])
                train_labels.append(data_y)
            elif file_name in tests:
                test_audio.append(out2.detach().numpy())
                test_inputs.append(encoded_dict['input_ids'])
                test_attention.append(encoded_dict['attention_mask'])
                test_labels.append(data_y)
            else:
                dev_audio.append(out2.detach().numpy())
                dev_inputs.append(encoded_dict['input_ids'])
                dev_attention.append(encoded_dict['attention_mask'])
                dev_labels.append(data_y)
        else:
            x_input = get_audio_vector(audio_feature, audio_model, file_name, dataset)
            out1, out2 = model(x_input.float())
            out2 = out2.squeeze()
            if counter in random_indexes:
                test_audio.append(out2.detach().numpy())
                test_inputs.append(encoded_dict['input_ids'])
                test_attention.append(encoded_dict['attention_mask'])
                test_labels.append(data_y)
            elif counter in val_indexes:
                dev_audio.append(out2.detach().numpy())
                dev_inputs.append(encoded_dict['input_ids'])
                dev_attention.append(encoded_dict['attention_mask'])
                dev_labels.append(data_y)
            else:
                train_audio.append(out2.detach().numpy())
                train_inputs.append(encoded_dict['input_ids'])
                train_attention.append(encoded_dict['attention_mask'])
                train_labels.append(data_y)
        counter += 1

    train_inputs = torch.cat(train_inputs, dim=0)
    train_attention = torch.cat(train_attention, dim=0)
    test_inputs = torch.cat(test_inputs, dim=0)
    test_attention = torch.cat(test_attention, dim=0)
    dev_inputs = torch.cat(dev_inputs, dim=0)
    dev_attention = torch.cat(dev_attention, dim=0)
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)
    dev_labels = torch.tensor(dev_labels)
    train_audio = torch.tensor(train_audio)
    test_audio = torch.tensor(test_audio)
    dev_audio = torch.tensor(dev_audio)
    return train_inputs, train_attention, train_labels, test_inputs, test_labels, \
           test_attention, dev_inputs, dev_attention, dev_labels, train_audio, test_audio, dev_audio


def bert_text_preparation(text, tokenizer):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return tokenized_text, tokens_tensor, segments_tensors


def get_bert_embeddings(tokens_tensor, segments_tensors, model):
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
    if dataset == "ECF":
        data, filtered_data = load_ecf_structure()
        for conv in data:
            for sentence in conv["conversation"]:
                words.extend(sentence["text"].lower().strip().split())
    else:
        iemocap_data = load_iemocap_structure()
        for entry in iemocap_data:
            words.extend(entry["text"].lower().strip().split())

    words = set(words)
    word_idx = dict((c, k + 1) for k, c in enumerate(words))

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


def get_audio_vector(audio_feature, model_name, filename, dataset):
    train_folder = "train"
    test_folder = "test"
    trains = os.listdir(train_folder)
    tests = os.listdir(test_folder)
    if dataset == "ECF":
        if filename in trains:
            audio_x, sample_rate = librosa.load("train/" + filename, duration=15)
        elif filename in tests:
            audio_x, sample_rate = librosa.load("test/" + filename, duration=15)
        else:
            audio_x, sample_rate = librosa.load("val/" + filename, duration=15)

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
        if model_name == "CNN2D":
            x_input = x_input[None, None, :, :]
        elif model_name == "CNN1D" or model_name == "LSTM":
            x_input = x_input[None, None, :]
        else:
            x_input = x_input[None, :]
    else:
        audio_x, sample_rate = librosa.load("IEMOCAP/audio/" + filename, duration=15)
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
        if model_name == "CNN2D":
            x_input = x_input[None, None, :, :]
        elif model_name == "CNN1D" or model_name == "LSTM":
            x_input = x_input[None, None, :]
        else:
            x_input = x_input[None, :]

    return x_input
