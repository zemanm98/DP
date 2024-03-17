import os
import nussl

def denoise():
    trains = os.listdir("train")
    tests = os.listdir("test")
    vals = os.listdir("val")
    for file_name in trains:
        signal1 = nussl.AudioSignal("train/" + file_name)
        # ft2d = nussl.separation.primitive.FT2D(signal1, mask_type='binary')
        # estimates = ft2d()
        separator = nussl.separation.primitive.RepetSim(signal1, mask_type='binary')
        estimates = separator()
        # estimates[0].write_audio_to_file("ft2d_background.wav")
        estimates[1].write_audio_to_file("train_REPETSIM/" + file_name)
    for file_name in tests:
        signal1 = nussl.AudioSignal("test/" + file_name)
        # ft2d = nussl.separation.primitive.FT2D(signal1, mask_type='binary')
        # estimates = ft2d()
        separator = nussl.separation.primitive.RepetSim(signal1, mask_type='binary')
        estimates = separator()
        # estimates[0].write_audio_to_file("ft2d_background.wav")
        estimates[1].write_audio_to_file("test_REPETSIM/" + file_name)
    for file_name in vals:
        signal1 = nussl.AudioSignal("val/" + file_name)
        # ft2d = nussl.separation.primitive.FT2D(signal1, mask_type='binary')
        # estimates = ft2d()
        separator = nussl.separation.primitive.RepetSim(signal1, mask_type='binary')
        estimates = separator()
        # estimates[0].write_audio_to_file("ft2d_background.wav")
        estimates[1].write_audio_to_file("val_REPETSIM/" + file_name)

        # y, sr = librosa.load("train/" + file_name, sr=None)
        # S_full, phase = librosa.magphase(librosa.stft(y))
        # idx = slice(*librosa.time_to_frames([30, 35], sr=sr))
        # plt.figure(figsize=(12, 4))
        # librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
        #                          y_axis='log', x_axis='time', sr=sr)
        # plt.colorbar()
        # plt.tight_layout()
        # S_filter = librosa.decompose.nn_filter(S_full,
        #                                        aggregate=np.median,
        #                                        metric='cosine',
        #                                        width=int(librosa.time_to_frames(2, sr=sr)))
        # S_filter = np.minimum(S_full, S_filter)
        # margin_i, margin_v = 2, 10
        # power = 2
        # mask_i = librosa.util.softmask(S_filter,
        #                                margin_i * (S_full - S_filter),
        #                                power=power)
        #
        # mask_v = librosa.util.softmask(S_full - S_filter,
        #                                margin_v * S_filter,
        #                                power=power)
        #
        # S_foreground = mask_v * S_full
        # S_background = mask_i * S_full
        # D_foreground = S_foreground * phase
        # D_background = S_background * phase
        # y_foreground = librosa.istft(D_foreground)
        # y_background = librosa.istft(D_background)
        # sf.write("train_denoised/" + file_name, y_foreground, samplerate=int(sr))
        # sf.write("background.wav", y_background, samplerate=int(sr))
        # X = librosa.stft(noisy_audio)
        # norbert.residual_model()
        # V = model(X)
        # Y = norbert.wiener(V, X)
        # estimate = istft(Y)

        # spectrum = np.abs(librosa.stft(noisy_audio))
        # spectrum = np.mean(spectrum, axis=1)
        # alpha = 0.6
        # clean_spectrum = np.maximum((spectrum - alpha) * spectrum, 0)
        # clean_aduio = librosa.istft(clean_spectrum)
