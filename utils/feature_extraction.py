import librosa
import numpy as np

def get_mfcconly_n_dim(data, sample_rate, pad_length=-1):
    mfc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=50)
    if pad_length > 0:
        if mfc.shape[1] < pad_length:
            append_vector = np.zeros([50, pad_length - mfc.shape[1]])
            mfc = np.append(mfc, append_vector, axis=1)
        elif mfc.shape[1] > pad_length:
            mfc = mfc[:, 0:pad_length]
    mfc = mfc.T
    return mfc


def get_collective_features(X, sample_rate) -> np.array:
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


def get_collective_features_n_dim(X, sample_rate, pad_length=-1) -> np.array:
    stft = np.abs(librosa.stft(X))
    pitches, magnitudes = librosa.piptrack(y=X, sr=sample_rate, S=stft, fmin=70, fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])

    cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
    cent = cent / np.sum(cent)

    flatness_n = librosa.feature.spectral_flatness(y=X)
    mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50)
    chroma_n = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    mel_n = librosa.feature.melspectrogram(y=X, sr=sample_rate)
    contrast_n = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
    zerocr_n = librosa.feature.zero_crossing_rate(X)

    S, phase = librosa.magphase(stft)

    rmse = librosa.feature.rms(S=S)[0]
    reshaped_pitch = np.reshape(np.array(pitch), (1, len(pitch)))
    reshaped_rmse = np.reshape(rmse, (1, len(rmse)))
    weird_vector = np.concatenate(
        (reshaped_pitch, cent, flatness_n, mfcc, chroma_n, mel_n, contrast_n, zerocr_n, reshaped_rmse)).T

    if pad_length > 0:
        if weird_vector.shape[0] < pad_length:
            append_vector = np.zeros([pad_length - weird_vector.shape[0], 202])
            weird_vector = np.append(weird_vector, append_vector, axis=0)
        elif weird_vector.shape[0] > pad_length:
            weird_vector = weird_vector[0:pad_length, :]

    return weird_vector


def get_mfcconly(data, sample_rate):
    mfc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=50).T, axis=0)
    return mfc