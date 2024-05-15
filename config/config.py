# LSTM multimodal or text model  hyperparameters
LSTM_TEXT_LR = 0.0001
LSTM_TEXT_BATCH_SIZE = 32

# BERT multimodal or text model hyperparameters
BERT_LR = 2e-5
BERT_BATCH_SIZE = 16

# CNN1D audio model hyperparameters
CNN1D_LR = 0.0001
CNN1D_BATCH_SIZE = 32

# CNN2D audio model hyperparameters
CNN2D_LR = 0.0001
CNN2D_BATCH_SIZE = 32

# NN audio model hyperparameters
NN_LR = 0.0001
NN_BATCH_SIZE = 32

#Number of Epochs for training
AUDIO_EPOCHS = 100
TEXT_AND_MULTIMODAL_EPOCHS_LSTM = 50
TEXT_AND_MULTIMODAL_EPOCHS_BERT = 3

# ECF dataset files path
ECF_TRAIN_FOLDER = ".\\data\\train"
ECF_TEST_FOLDER = ".\\data\\test"
ECF_DEV_FOLDER = ".\\data\\val"
ECF_TEXT_JSON_PATH = ".\\data\\Subtask_2_2_train.json"

# IEMOCAP dataset files path
IEMOCAP_TRANSCRIPTIONS_FODER = ".\\data\\IEMOCAP\\transcription"
IEMOCAP_EMOTIONS_FOLDER = ".\\data\\IEMOCAP\\emotions"
IEMOCAP_AUDIO_FOLDER = ".\\data\\IEMOCAP\\audio"

# RAVDESS dataset files path
RAVDESS_PATH = ".\\data\\RAVDESS"

# Word2Vec embeddings file
W2V_FILE_PATH = ".\\data\\w2v\\model.txt"
