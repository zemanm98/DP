import torch
import os
import numpy as np
import random
import librosa
import json

import torchaudio
from transformers import BertTokenizer, BertModel
from transformers import AutoFeatureExtractor
from models import ConvNet, LSTM_extractor_old

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
f = open("data/Subtask_2_2_train.json")
data = json.load(f)
filtered_data = {}
speakers_data = {}
# speakers_occurences = {}
for h in data:
    for g in h["conversation"]:
        filtered_data[str(h["conversation_ID"]) + "_" + str(g["utterance_ID"])] = g["emotion"]
        speakers_data[str(h["conversation_ID"]) + "_" + str(g["utterance_ID"])] = g["speaker"]
        # if g["speaker"] not in speakers_occurences:
        #     speakers_occurences[g["speaker"]] = 0
        # speakers_occurences[g["speaker"]] += 1


# speakers_occurences = sorted(speakers_occurences.items(), key=lambda x:x[1], reverse=True)

def get_features_ConvNet(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    return result


def get_features_mfcconly(data, sample_rate, pad_length=-1):
    mfc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=50)
    if pad_length > 0:
        if mfc.shape[1] < pad_length:
            append_vector = np.zeros([50, pad_length - mfc.shape[1]])
            mfc = np.append(mfc, append_vector, axis=1)
        elif mfc.shape[1] > pad_length:
            mfc = mfc[:, 0:pad_length]
    mfc = mfc.T
    return mfc


def get_features_ndim(x, sample_rate, pad_length):
    hop_length = int(0.10 * sample_rate)
    n_fft = int(0.20 * sample_rate)
    mfc = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
    if mfc.shape[1] < 7:
        return None
    mfc = mfc[1:, :]
    mfc_delta = librosa.feature.delta(mfc, width=7)
    mfc_delta_delta = librosa.feature.delta(mfc, width=7, order=2)
    features = np.concatenate((mfc, mfc_delta, mfc_delta_delta), axis=0)
    features = np.swapaxes(features, 0, 1)
    if pad_length > 0:
        if features.shape[0] < pad_length:
            append_vector = np.zeros([pad_length - features.shape[0], 36])
            features = np.append(features, append_vector, axis=0)
        elif features.shape[0] > pad_length:
            features = features[0:pad_length, :]
    return features


def get_features(X, sample_rate, pad_length=-1) -> np.array:
    stft = np.abs(librosa.stft(X))
    pitches, magnitudes = librosa.piptrack(y=X, sr=sample_rate, S=stft, fmin=70, fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])

    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    pitchmean = np.mean(pitch)
    pitchstd = np.std(pitch)
    pitchmax = np.max(pitch)
    pitchmin = np.min(pitch)

    cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
    cent = cent / np.sum(cent)
    meancent = np.mean(cent)
    stdcent = np.std(cent)
    maxcent = np.max(cent)

    # flatness_n = librosa.feature.spectral_flatness(y=X)
    flatness = np.mean(librosa.feature.spectral_flatness(y=X))

    # mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccsstd = np.std(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccmax = np.max(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)

    # chroma_n = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # mel_n = librosa.feature.melspectrogram(y=X, sr=sample_rate)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)

    # contrast_n = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    # zerocr_n = librosa.feature.zero_crossing_rate(X)
    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

    S, phase = librosa.magphase(stft)
    meanMagnitude = np.mean(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)

    rmse = librosa.feature.rms(S=S)[0]
    meanrms = np.mean(rmse)
    stdrms = np.std(rmse)
    maxrms = np.max(rmse)
    # reshaped_pitch = np.reshape(np.array(pitch), (1, len(pitch)))
    # reshaped_rmse = np.reshape(rmse, (1, len(rmse)))
    # weird_vector = np.concatenate(
    #     (reshaped_pitch, cent, flatness_n, mfcc, chroma_n, mel_n, contrast_n, zerocr_n, reshaped_rmse)).T
    ext_features = np.array([
        flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
        maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
        pitch_tuning_offset, meanrms, maxrms, stdrms
    ])
    # if pad_length > 0:
    #     if weird_vector.shape[0] < pad_length:
    #         append_vector = np.zeros([pad_length - weird_vector.shape[0], 202])
    #         weird_vector = np.append(weird_vector, append_vector, axis=0)
    #     elif weird_vector.shape[0] > pad_length:
    #         weird_vector = weird_vector[0:pad_length, :]
    ext_features = np.concatenate((ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast))

    return ext_features


def get_mfcc_flat(data, sample_rate, pad_length=-1):
    mfc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=50).T, axis=0)
    return mfc


def load_IEMOCAP():
    iemocap_data = torchaudio.datasets.IEMOCAP(".")
    for q in iemocap_data:
        print("a")


def load_RAVDESS():
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
            audio_x, sample_rate = librosa.load("RAVDESS/" + f + "/" + file, duration=10)
            # hop_length = int(0.10 * sample_rate)
            # n_fft = int(0.20 * sample_rate)
            # mfc = librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
            # mfc = mfc[1:, :]
            # mfc_delta = librosa.feature.delta(mfc, width=7)
            # mfc_delta_delta = librosa.feature.delta(mfc, width=7, order=2)
            # features = np.concatenate((mfc, mfc_delta, mfc_delta_delta), axis=0)
            # features = np.swapaxes(features, 0, 1)
            # features = get_features(audio_x, sample_rate)
            features = get_features(audio_x, sample_rate, 300)
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
    return train_x, val_x, train_y, val_y, [], [], test_x, test_y, []


def load_dataset():
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
    train_speakers = []
    val_speakers = []
    test_x = []
    test_y = []
    test_speakers = []
    max_length = 0
    for f in trains:
        key = f[3:].split("utt")[0] + "_" + f[3:].split("utt")[1].split(".")[0]
        y = np.zeros(7)
        y_speakers = np.zeros(7)
        y[emotion_categories.index(filtered_data[key])] = 1.0
        speaker = speakers_data[key]
        if speaker not in speakers:
            y_speakers[6] = 1.0
        else:
            y_speakers[speakers.index(speaker)] = 1.0
        audio_x, sample_rate = librosa.load("train/" + f, duration=15)

        # features = get_features_mfcconly(audio_x, sample_rate, 400)
        features = get_features(audio_x, sample_rate, 300)

        if features is None:
            continue
        # mfc = np.transpose(mfc)
        # mfc -= (np.mean(mfc, axis=0) + 1e-8)
        # features = np.mean(librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=50).T, axis=0)
        train_y.append(y)
        train.append(features)
        train_speakers.append(y_speakers)
    for f in vals:
        key = f[3:].split("utt")[0] + "_" + f[3:].split("utt")[1].split(".")[0]
        y = np.zeros(7)
        y_speakers = np.zeros(7)
        y[emotion_categories.index(filtered_data[key])] = 1.0
        speaker = speakers_data[key]
        if speaker not in speakers:
            y_speakers[6] = 1.0
        else:
            y_speakers[speakers.index(speaker)] = 1.0
        audio_x, sample_rate = librosa.load("val/" + f, duration=15)

        # features = get_features_mfcconly(audio_x, sample_rate, 400)
        features = get_features(audio_x, sample_rate, 300)

        if features is None:
            continue
        # mfc = np.transpose(mfc)
        # mfc -= (np.mean(mfc, axis=0) + 1e-8)
        # features = np.mean(librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=50).T, axis=0)
        val.append(features)
        val_y.append(y)
        val_speakers.append(y_speakers)
    for f in tests:
        key = f[3:].split("utt")[0] + "_" + f[3:].split("utt")[1].split(".")[0]
        y = np.zeros(7)
        y_speakers = np.zeros(7)
        y[emotion_categories.index(filtered_data[key])] = 1.0
        speaker = speakers_data[key]
        if speaker not in speakers:
            y_speakers[6] = 1.0
        else:
            y_speakers[speakers.index(speaker)] = 1.0
        audio_x, sample_rate = librosa.load("test/" + f, duration=15)

        # features = get_features_mfcconly(audio_x, sample_rate, 400)
        features = get_features(audio_x, sample_rate, 300)

        if features is None:
            continue
        # mfc = np.transpose(mfc)
        # mfc -= (np.mean(mfc, axis=0) + 1e-8)
        # features = np.mean(librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=50).T, axis=0)
        test_x.append(features)
        test_y.append(y)
        test_speakers.append(y_speakers)

    # train, val, train_y, val_y, train_speakers, val_speakers, test_x, test_y, test_speakers =\
    #     map(torch.tensor, [train, val, train_y, val_y, train_speakers, val_speakers, test_x, test_y, test_speakers])
    val = [torch.from_numpy(i) for i in val]
    val = torch.stack((val))
    test_x = [torch.from_numpy(i) for i in test_x]
    test_x = torch.stack((test_x))
    # val = pad_sequence(val, batch_first=True)
    # test_x = pad_sequence(test_x, batch_first=True)
    train_y, val_y, train_speakers, val_speakers, test_y, test_speakers = \
        map(torch.tensor, [train_y, val_y, train_speakers, val_speakers, test_y, test_speakers])
    return train, val.float(), train_y.float(), val_y.float(), train_speakers.float(), val_speakers.float(), test_x.float(), \
           test_y.float(), test_speakers.float()


def load_old_dataset():
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
    train_speakers = []
    val_speakers = []
    test_x = []
    test_y = []
    test_speakers = []
    for f in trains:
        key = f[3:].split("utt")[0] + "_" + f[3:].split("utt")[1].split(".")[0]
        y = np.zeros(7)
        y_speakers = np.zeros(7)
        y[emotion_categories.index(filtered_data[key])] = 1.0
        speaker = speakers_data[key]
        if speaker not in speakers:
            y_speakers[6] = 1.0
        else:
            y_speakers[speakers.index(speaker)] = 1.0
        audio_x, sample_rate = librosa.load("train/" + f, duration=10)
        # features = np.mean(librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=50).T, axis=0)
        features = get_features(audio_x, sample_rate, 300)
        train_y.append(y)
        train.append(features)
        train_speakers.append(y_speakers)
    for f in vals:
        key = f[3:].split("utt")[0] + "_" + f[3:].split("utt")[1].split(".")[0]
        y = np.zeros(7)
        y_speakers = np.zeros(7)
        y[emotion_categories.index(filtered_data[key])] = 1.0
        speaker = speakers_data[key]
        if speaker not in speakers:
            y_speakers[6] = 1.0
        else:
            y_speakers[speakers.index(speaker)] = 1.0
        audio_x, sample_rate = librosa.load("val/" + f, duration=10)
        # features = np.mean(librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=50).T, axis=0)
        features = get_features(audio_x, sample_rate, 300)
        val.append(features)
        val_y.append(y)
        val_speakers.append(y_speakers)
    for f in tests:
        key = f[3:].split("utt")[0] + "_" + f[3:].split("utt")[1].split(".")[0]
        y = np.zeros(7)
        y_speakers = np.zeros(7)
        y[emotion_categories.index(filtered_data[key])] = 1.0
        speaker = speakers_data[key]
        if speaker not in speakers:
            y_speakers[6] = 1.0
        else:
            y_speakers[speakers.index(speaker)] = 1.0
        audio_x, sample_rate = librosa.load("test/" + f, duration=10)
        # features = np.mean(librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=50).T, axis=0)
        features = get_features(audio_x, sample_rate, 300)
        test_x.append(features)
        test_y.append(y)
        test_speakers.append(y_speakers)

    train, train_y, val_y, train_speakers, val_speakers, test_y, test_speakers, val, test_x = \
        map(torch.tensor, [train, train_y, val_y, train_speakers, val_speakers, test_y, test_speakers, val, test_x])
    return train.float(), val.float(), train_y.float(), val_y.float(), train_speakers.float(), val_speakers.float(), \
           test_x.float(), test_y.float(), test_speakers.float()


def load_wav2vec():
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
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
    train_speakers = []
    val_speakers = []
    test_x = []
    test_y = []
    test_speakers = []
    for f in trains:
        key = f[3:].split("utt")[0] + "_" + f[3:].split("utt")[1].split(".")[0]
        print(key)
        y = np.zeros(7)
        y_speakers = np.zeros(7)
        y[emotion_categories.index(filtered_data[key])] = 1.0
        speaker = speakers_data[key]
        if speaker not in speakers:
            y_speakers[6] = 1.0
        else:
            y_speakers[speakers.index(speaker)] = 1.0
        train_y.append(y)
        train_speakers.append(y_speakers)
        audio_x, sample_rate = librosa.load("train/" + f)
        speech = librosa.resample(audio_x, orig_sr=sample_rate, target_sr=16000)
        inputs = feature_extractor(speech, truncation=True, max_length=50000, padding='max_length', sampling_rate=16000,
                                   return_tensors="np").input_values
        inputs = inputs.flatten()
        train.append(inputs)
    for f in vals:
        key = f[3:].split("utt")[0] + "_" + f[3:].split("utt")[1].split(".")[0]
        print(key)
        y = np.zeros(7)
        y_speakers = np.zeros(7)
        y[emotion_categories.index(filtered_data[key])] = 1.0
        speaker = speakers_data[key]
        if speaker not in speakers:
            y_speakers[6] = 1.0
        else:
            y_speakers[speakers.index(speaker)] = 1.0
        val_y.append(y)
        val_speakers.append(y_speakers)
        audio_x, sample_rate = librosa.load("val/" + f)
        speech = librosa.resample(audio_x, orig_sr=sample_rate, target_sr=16000)
        inputs = feature_extractor(speech, truncation=True, max_length=50000, padding='max_length', sampling_rate=16000,
                                   return_tensors="np").input_values
        inputs = inputs.flatten()
        val.append(inputs)
    for f in tests:
        key = f[3:].split("utt")[0] + "_" + f[3:].split("utt")[1].split(".")[0]
        print(key)
        y = np.zeros(7)
        y_speakers = np.zeros(7)
        y[emotion_categories.index(filtered_data[key])] = 1.0
        speaker = speakers_data[key]
        if speaker not in speakers:
            y_speakers[6] = 1.0
        else:
            y_speakers[speakers.index(speaker)] = 1.0
        test_y.append(y)
        test_speakers.append(y_speakers)
        audio_x, sample_rate = librosa.load("test/" + f)
        speech = librosa.resample(audio_x, orig_sr=sample_rate, target_sr=16000)
        inputs = feature_extractor(speech, truncation=True, max_length=50000, padding='max_length', sampling_rate=16000,
                                   return_tensors="np").input_values
        inputs = inputs.flatten()
        test_x.append(inputs)

    train, val, train_y, val_y, train_speakers, val_speakers, test_x, test_y, test_speakers = \
        map(torch.tensor, [train, val, train_y, val_y, train_speakers, val_speakers, test_x, test_y, test_speakers])
    return train, val, train_y, val_y, train_speakers, val_speakers, test_x, test_y, test_speakers


def load_text_data(word_idx, max_doc_len, max_sen_len, n_class):
    train_folder = "train"
    val_folder = "val"
    test_folder = "test"
    trains = os.listdir(train_folder)
    vals = os.listdir(val_folder)
    tests = os.listdir(test_folder)
    train_x, train_y, test_x, test_y, dev_x, dev_y, train_a, test_a, dev_a = [], [], [], [], [], [], [], [], []
    model = LSTM_extractor_old()
    model.load_state_dict(torch.load("./mfcc_model/LSTM_ecf_collective_features.pt"))
    for conv in data:
        d_len = len(conv["conversation"])
        # data_x, data_y = np.zeros((max_doc_len, max_sen_len)), np.zeros((max_doc_len, n_class))
        for i in range(d_len):
            data_x, data_y = np.zeros(max_sen_len), np.zeros(n_class)
            data_y[emotion_categories.index(conv["conversation"][i]["emotion"])] = 1
            line = conv["conversation"][i]["text"].strip().split()
            for j, word in enumerate(line):
                word = word.lower()
                if j >= max_sen_len:
                    break
                # elif word not in word_idx : data_x[i][j] = 24166
                # else : data_x[i][j] = int(word_idx[word])
                elif word not in word_idx:
                    data_x[j] = 6434
                else:
                    data_x[j] = int(word_idx[word])

            file_name = conv["conversation"][0]["video_name"][:-3] + "wav"
            if file_name in trains:
                audio_x, sample_rate = librosa.load("train/" + file_name, duration=15)
                audio_data = get_features(audio_x, sample_rate, 300)
                x_input = torch.from_numpy(audio_data)
                x_input = x_input[None, None, :]
                out1, out2 = model(x_input.float())
                out2 = out2.squeeze()
                train_a.append(out2.detach().numpy())
                train_x.append(data_x)
                train_y.append(data_y)
            elif file_name in tests:
                audio_x, sample_rate = librosa.load("test/" + file_name, duration=15)
                audio_data = get_features(audio_x, sample_rate, 300)
                x_input = torch.from_numpy(audio_data)
                x_input = x_input[None, None, :]
                out1, out2 = model(x_input.float())
                out2 = out2.squeeze()
                test_a.append(out2.detach().numpy())
                test_x.append(data_x)
                test_y.append(data_y)
            else:
                audio_x, sample_rate = librosa.load("val/" + file_name, duration=15)
                audio_data = get_features(audio_x, sample_rate, 300)
                x_input = torch.from_numpy(audio_data)
                x_input = x_input[None, None, :]
                out1, out2 = model(x_input.float())
                out2 = out2.squeeze()
                dev_a.append(out2.detach().numpy())
                dev_x.append(data_x)
                dev_y.append(data_y)

    train_a = torch.tensor(train_a)
    test_a = torch.tensor(test_a)
    dev_a = torch.tensor(dev_a)
    train_x, train_y, test_x, test_y, dev_x, dev_y = map(torch.tensor,
                                                         [train_x, train_y, test_x, test_y, dev_x, dev_y])

    return train_x, train_y, test_x, test_y, dev_x, dev_y, train_a, test_a, dev_a


def load_text_data_bert(max_sen_len, n_class):
    train_folder = "train"
    val_folder = "val"
    test_folder = "test"
    bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True, )
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    trains = os.listdir(train_folder)
    vals = os.listdir(val_folder)
    tests = os.listdir(test_folder)
    train_x, train_y, test_x, test_y, dev_x, dev_y = [], [], [], [], [], []
    for conv in data:
        d_len = len(conv["conversation"])
        # data_x, data_y = np.zeros((max_doc_len, max_sen_len)), np.zeros((max_doc_len, n_class))
        for i in range(d_len):
            data_x, data_y = np.zeros((max_sen_len, 768)), np.zeros(n_class)
            data_y[emotion_categories.index(conv["conversation"][i]["emotion"])] = 1
            line = conv["conversation"][i]["text"].strip()
            tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(line, bert_tokenizer)
            embedding = get_bert_embeddings(tokens_tensor, segments_tensors, bert_model)
            for j, emb in enumerate(embedding):
                if j >= max_sen_len:
                    break
                data_x[j] = np.array(emb)

            file_name = conv["conversation"][0]["video_name"][:-3] + "wav"
            if file_name in trains:
                train_x.append(data_x)
                train_y.append(data_y)
            elif file_name in tests:
                test_x.append(data_x)
                test_y.append(data_y)
            else:
                dev_x.append(data_x)
                dev_y.append(data_y)

    train_x, train_y, test_x, test_y, dev_x, dev_y = map(torch.tensor,
                                                         [train_x, train_y, test_x, test_y, dev_x, dev_y])

    return train_x, train_y, test_x, test_y, dev_x, dev_y


def load_data_for_bert(max_sen_len, n_class):
    train_folder = "train"
    val_folder = "val"
    test_folder = "test"
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    trains = os.listdir(train_folder)
    vals = os.listdir(val_folder)
    tests = os.listdir(test_folder)
    train_inputs, train_attention, train_labels, test_inputs, test_labels, \
    test_attention, dev_inputs, dev_attention, dev_labels, train_audio, test_audio, dev_audio = [], [], [], [], [], [], \
                                                                                                [], [], [], [], [], []
    model = ConvNet()
    model.load_state_dict(torch.load("./mfcc_model/CNN1D_efc.pt"))
    max_sen_len = 91
    for conv in data:
        d_len = len(conv["conversation"])
        # data_x, data_y = np.zeros((max_doc_len, max_sen_len)), np.zeros((max_doc_len, n_class))
        for i in range(d_len):
            data_y = np.zeros(n_class)
            line = conv["conversation"][i]["text"].strip()
            data_y[emotion_categories.index(conv["conversation"][i]["emotion"])] = 1
            encoded_dict = bert_tokenizer.encode_plus(
                line,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=max_sen_len,  # Pad & truncate all sentences.
                padding='max_length',
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )
            # if encoded_dict['input_ids'].size()[1] > max_length:
            #     max_length = encoded_dict['input_ids'].size()[1]
            file_name = conv["conversation"][0]["video_name"][:-3] + "wav"
            if file_name in trains:
                audio_x, sample_rate = librosa.load("train/" + file_name, duration=15)
                audio_data = get_features(audio_x, sample_rate, 300)
                x_input = torch.from_numpy(audio_data)
                x_input = x_input[None, None, :]
                out1, out2 = model(x_input.float())
                out2 = out2.squeeze()
                train_inputs.append(encoded_dict['input_ids'])
                train_attention.append(encoded_dict['attention_mask'])
                train_labels.append(data_y)
                train_audio.append(out2.detach().numpy())
            elif file_name in tests:
                audio_x, sample_rate = librosa.load("test/" + file_name, duration=15)
                audio_data = get_features(audio_x, sample_rate, 300)
                x_input = torch.from_numpy(audio_data)
                x_input = x_input[None, None, :]
                out1, out2 = model(x_input.float())
                out2 = out2.squeeze()
                test_inputs.append(encoded_dict['input_ids'])
                test_attention.append(encoded_dict['attention_mask'])
                test_labels.append(data_y)
                test_audio.append(out2.detach().numpy())
            else:
                audio_x, sample_rate = librosa.load("val/" + file_name, duration=15)
                audio_data = get_features(audio_x, sample_rate, 300)
                x_input = torch.from_numpy(audio_data)
                x_input = x_input[None, None, :]
                out1, out2 = model(x_input.float())
                out2 = out2.squeeze()
                dev_inputs.append(encoded_dict['input_ids'])
                dev_attention.append(encoded_dict['attention_mask'])
                dev_labels.append(data_y)
                dev_audio.append(out2.detach().numpy())

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


def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nloading embedding vectors\n')

    words = []
    inputFile = open(train_file_path, 'r', encoding="utf-8")
    for line in inputFile.readlines():
        line = line.strip().split(';')
        emotion, clause = line[1], line[-1]
        words.extend(emotion.lower().split() + clause.lower().split())

    words = set(words)
    word_idx = dict((c, k + 1) for k, c in enumerate(words))
    # word_idx_rev = dict((k + 1, c) for k, c in enumerate(words))

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

    # embedding_pos = [list(np.zeros(embedding_dim_pos))]
    # embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) \
    #                       for i in range(200)])
    # embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)
    embedding = np.array(embedding)

    # print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("embedding.shape: {}".format(embedding.shape))
    print("load embedding done!\n")
    # return word_idx_rev, word_idx, embedding, embedding_pos
    return word_idx, embedding
