import torch
import os
import numpy as np
import random
import librosa
import json
from transformers import AutoFeatureExtractor

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

def get_features_CNN(data, sample_rate):
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

def get_features(X, sample_rate) -> np.array:
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

    flatness = np.mean(librosa.feature.spectral_flatness(y=X))

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccsstd = np.std(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccmax = np.max(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

    S, phase = librosa.magphase(stft)
    meanMagnitude = np.mean(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)

    rmse = librosa.feature.rms(S=S)[0]
    meanrms = np.mean(rmse)
    stdrms = np.std(rmse)
    maxrms = np.max(rmse)

    ext_features = np.array([
        flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
        maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
        pitch_tuning_offset, meanrms, maxrms, stdrms
    ])

    ext_features = np.concatenate((ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast))

    return ext_features

def load_RAVDESS():
    folders = os.listdir("RAVDESS")
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    random_indexes = []
    while len(random_indexes) < 250:
        index = random.randint(0, 1399)
        if index not in random_indexes:
            random_indexes.append(index)
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
            features = get_features_CNN(audio_x, sample_rate)
            if counter in random_indexes:
                test_x.append(features)
                test_y.append(y)
            else:
                train_x.append(features)
                train_y.append(y)
            counter += 1
    train_y = [torch.from_numpy(train_y[i]) for i in range(0, len(train_y))]
    train_y = torch.stack(train_y)
    test_y = [torch.from_numpy(test_y[i]) for i in range(0, len(test_y))]
    test_y = torch.stack(test_y)
    train_x, train_y, test_x, test_y = map(torch.tensor, [train_x, train_y, test_x, test_y])
    train_x = train_x.float()
    train_y = train_y.float()
    test_x = test_x.float()
    test_y = test_y.float()
    return train_x, [], train_y, [], [], [], test_x, test_y, []

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
        key = f[3:].split("utt")[0] + "_" +f[3:].split("utt")[1].split(".")[0]
        y = np.zeros(7)
        y_speakers = np.zeros(7)
        y[emotion_categories.index(filtered_data[key])] = 1.0
        speaker = speakers_data[key]
        if speaker not in speakers:
            y_speakers[6] = 1.0
        else:
            y_speakers[speakers.index(speaker)] = 1.0
        audio_x, sample_rate = librosa.load("train/" + f, duration=10)

        hop_length = int(0.10 * sample_rate)
        n_fft = int(0.20 * sample_rate)
        mfc = librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
        if mfc.shape[1] < 7:
            continue
        mfc = mfc[1:, :]
        mfc_delta = librosa.feature.delta(mfc, width=7)
        mfc_delta_delta = librosa.feature.delta(mfc, width=7, order=2)
        features = np.concatenate((mfc, mfc_delta, mfc_delta_delta), axis=0)
        features = np.swapaxes(features, 0, 1)
        if features.shape[0] < 80:
            append_vector = np.zeros([80 - features.shape[0], 36])
            features = np.append(features, append_vector, axis=0)
        elif features.shape[0] > 80:
            features = features[0:80, :]
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
        audio_x, sample_rate = librosa.load("val/" + f, duration=10)
        hop_length = int(0.10 * sample_rate)
        n_fft = int(0.20 * sample_rate)
        mfc = librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
        if mfc.shape[1] < 7:
            continue
        mfc = mfc[1:, :]
        mfc_delta = librosa.feature.delta(mfc, width=7)
        mfc_delta_delta = librosa.feature.delta(mfc, width=7, order=2)
        features = np.concatenate((mfc, mfc_delta, mfc_delta_delta), axis=0)
        features = np.swapaxes(features, 0, 1)
        if features.shape[0] < 80:
            append_vector = np.zeros([80 - features.shape[0], 36])
            features = np.append(features, append_vector, axis=0)
        elif features.shape[0] > 80:
            features = features[0:80, :]
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
        audio_x, sample_rate = librosa.load("test/" + f, duration=10)
        hop_length = int(0.10 * sample_rate)
        n_fft = int(0.20 * sample_rate)
        # mfc = librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=40, hop_length=hop_length, n_fft=n_fft)
        mfc = librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
        if mfc.shape[1] < 7:
            continue
        mfc = mfc[1:, :]
        mfc_delta = librosa.feature.delta(mfc, width=7)
        mfc_delta_delta = librosa.feature.delta(mfc, width=7, order=2)
        features = np.concatenate((mfc, mfc_delta, mfc_delta_delta), axis=0)
        features = np.swapaxes(features, 0, 1)
        if features.shape[0] < 80:
            append_vector = np.zeros([80 - features.shape[0], 36])
            features = np.append(features, append_vector, axis=0)
        elif features.shape[0] > 80:
            features = features[0:80, :]
        # mfc = np.transpose(mfc)
        # mfc -= (np.mean(mfc, axis=0) + 1e-8)
        # features = np.mean(librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=50).T, axis=0)
        test_x.append(features)
        test_y.append(y)
        test_speakers.append(y_speakers)

    # train, val, train_y, val_y, train_speakers, val_speakers, test_x, test_y, test_speakers =\
    #     map(torch.tensor, [train, val, train_y, val_y, train_speakers, val_speakers, test_x, test_y, test_speakers])
    val = [torch.from_numpy(i) for i in val]
    test_x = [torch.from_numpy(i) for i in test_x]
    # val = pad_sequence(val, batch_first=True)
    # test_x = pad_sequence(test_x, batch_first=True)
    train_y, val_y, train_speakers, val_speakers, test_y, test_speakers = \
        map(torch.tensor, [train_y, val_y, train_speakers, val_speakers, test_y, test_speakers])
    return train, val, train_y.float(), val_y.float(), train_speakers.float(), val_speakers.float(), test_x,\
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
        key = f[3:].split("utt")[0] + "_" +f[3:].split("utt")[1].split(".")[0]
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
        features = get_features_CNN(audio_x, sample_rate)
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
        features = get_features_CNN(audio_x, sample_rate)
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
        features = get_features_CNN(audio_x, sample_rate)
        test_x.append(features)
        test_y.append(y)
        test_speakers.append(y_speakers)

    train, train_y, val_y, train_speakers, val_speakers, test_y, test_speakers, val, test_x = \
        map(torch.tensor, [train, train_y, val_y, train_speakers, val_speakers, test_y, test_speakers, val, test_x])
    return train.float(), val.float(), train_y.float(), val_y.float(), train_speakers.float(), val_speakers.float(),\
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
        inputs = feature_extractor(speech, truncation=True, max_length=50000, padding='max_length', sampling_rate=16000, return_tensors="np").input_values
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
        inputs = feature_extractor(speech, truncation=True, max_length=50000, padding='max_length', sampling_rate=16000, return_tensors="np").input_values
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
        inputs = feature_extractor(speech, truncation=True, max_length=50000, padding='max_length', sampling_rate=16000, return_tensors="np").input_values
        inputs = inputs.flatten()
        test_x.append(inputs)

    train, val, train_y, val_y, train_speakers, val_speakers, test_x, test_y, test_speakers = \
        map(torch.tensor, [train, val, train_y, val_y, train_speakers, val_speakers, test_x, test_y, test_speakers])
    return train, val, train_y, val_y, train_speakers, val_speakers, test_x, test_y, test_speakers