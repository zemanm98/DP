import librosa
import numpy as np

def get_mfcconly_n_dim(data, sample_rate, pad_length=-1):
    '''
    Returns mfcc feature matrix. Depending on the pad_length pads the output to desired size.
    '''
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
    '''
    Returns 1D feature vector of collective feature extraction methods.
    '''
    # performs the short time fourier transform on the input signal
    stft = np.abs(librosa.stft(X))
    # recognizes the pitch and magnitude values
    pitches, magnitudes = librosa.piptrack(y=X, sr=sample_rate, S=stft, fmin=70, fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])

    # estimates the offset of the pitches in input signal
    pitch_tuning_offset = librosa.pitch_tuning(pitches)

    pitchmean = np.mean(pitch)
    pitchstd = np.std(pitch)
    pitchmax = np.max(pitch)

    # retrieves the spectral centroid vector as a feature vector
    cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
    # normalize the spectral centroid features
    cent = cent / np.sum(cent)
    meancent = np.mean(cent)
    stdcent = np.std(cent)
    maxcent = np.max(cent)

    # retrieves the spectral flatness feature vector
    flatness = np.mean(librosa.feature.spectral_flatness(y=X))

    # retrieves the mfcc vector. Takes the mean, standart deviation and max over the time series.
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccsstd = np.std(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccmax = np.max(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)

    # retrieves the chromagram feature vector
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # retrieves the mel frequency spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)

    # retrieves the spectral contrast feature vector
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    # retrieves the zero crossing rate feature vector
    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

    # separates the signal into magnitude (S) and phase
    S, phase = librosa.magphase(stft)
    meanMagnitude = np.mean(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)

    # Root mean square value for each frame.
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
    '''
    Similar to the  get_collective_features method. Creates the collective feature vector with the time
    series dimension. Used for CNN2D audio emotion recognition model.
    '''
    # performs the short time fourier transform on the input signal
    stft = np.abs(librosa.stft(X))
    # recognizes the pitch and magnitude values
    pitches, magnitudes = librosa.piptrack(y=X, sr=sample_rate, S=stft, fmin=70, fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])

    # retrieves the spectral centroid vector as a feature vector
    cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
    # normalize the spectral centroid features
    cent = cent / np.sum(cent)

    # gets the spectral flatness features
    flatness_n = librosa.feature.spectral_flatness(y=X)

    # gets the mfc coefficients
    mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50)

    # gets the chromagaram features
    chroma_n = librosa.feature.chroma_stft(S=stft, sr=sample_rate)

    # gets the mel frequency spectrogram features
    mel_n = librosa.feature.melspectrogram(y=X, sr=sample_rate)

    # gets the spectral contrast features
    contrast_n = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)

    # gets the zero crossing rate features
    zerocr_n = librosa.feature.zero_crossing_rate(X)

    # separates the signal into magnitude (S) and phase
    S, phase = librosa.magphase(stft)

    # Root mean square value for each frame.
    rmse = librosa.feature.rms(S=S)[0]

    # reshapes the pitch and rms feature vectors
    reshaped_pitch = np.reshape(np.array(pitch), (1, len(pitch)))
    reshaped_rmse = np.reshape(rmse, (1, len(rmse)))
    weird_vector = np.concatenate(
        (reshaped_pitch, cent, flatness_n, mfcc, chroma_n, mel_n, contrast_n, zerocr_n, reshaped_rmse)).T

    # pads zero vectors  to the defined length
    if pad_length > 0:
        if weird_vector.shape[0] < pad_length:
            append_vector = np.zeros([pad_length - weird_vector.shape[0], 202])
            weird_vector = np.append(weird_vector, append_vector, axis=0)
        elif weird_vector.shape[0] > pad_length:
            weird_vector = weird_vector[0:pad_length, :]

    return weird_vector


def get_mfcconly(data, sample_rate):
    '''
    Returns mfcc feature vector where it takes the average values over the time series for each coefficient
    '''
    mfc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=50).T, axis=0)
    return mfc