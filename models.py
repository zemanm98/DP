import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel

'''
LSTM audio emotion recognition model
'''
class LSTM(torch.nn.Module):
    def __init__(self, feature_method, dataset):
        super(LSTM, self).__init__()
        # input size of the first LSTM layer depends on the feature extaction method used.
        if feature_method == "collective_features":
            self.bilstm = torch.nn.LSTM(312, 300, batch_first=True, bidirectional=True)
        else:
            self.bilstm = torch.nn.LSTM(50, 300, batch_first=True, bidirectional=True)
        self.dp = torch.nn.Dropout(p=0.3)
        self.linear = torch.nn.Linear(600, 50)
        self.relu = torch.nn.ReLU()
        # depending on the dataset used the output classification vector has different size
        if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
            self.linear2 = torch.nn.Linear(50, 7)
        elif dataset == "RAVDESS":
            self.linear2 = torch.nn.Linear(50, 8)
        else:
            self.linear2 = torch.nn.Linear(50, 5)

    def forward(self, x):
        x_states, hidden_states = self.bilstm(x)
        x = x_states.squeeze()
        x = self.dp(x)
        x_out = self.linear(x)
        x = self.relu(x_out)
        x = self.linear2(x)
        x = F.softmax(x)
        return x, x_out


'''
CNN1D audio emotion recognition model
'''
class ConvNet(torch.nn.Module):
    def __init__(self, feature_method, dataset):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 256, kernel_size=5, stride=1, padding=2)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2)
        self.pool2 = torch.nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.conv3 = torch.nn.Conv1d(256, 128, kernel_size=5, stride=1, padding=2)
        self.pool3 = torch.nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.dropout = torch.nn.Dropout(0.3)
        self.conv4 = torch.nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2)
        self.pool4 = torch.nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.flatten = torch.nn.Flatten()
        # the input size of the first linear layer changes based on the feature extraction method used.
        if feature_method == "collective_features":
            self.fc1 = torch.nn.Linear(1280, 512)
        else:
            self.fc1 = torch.nn.Linear(256, 512)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(512, 50)
        # the output classification vector depends on the dataset
        if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
            self.fc3 = torch.nn.Linear(50, 7)
        elif dataset == "RAVDESS":
            self.fc3 = torch.nn.Linear(50, 8)
        else:
            self.fc3 = torch.nn.Linear(50, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x_out = self.fc2(x)
        x = self.dropout2(x_out)
        x = self.fc3(x)
        return x, x_out


'''
CNN2D audio emotion recognition model
'''
class CNN_small(torch.nn.Module):
    def __init__(self, feature_method, dataset):
        super(CNN_small, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, kernel_size=(3, 1))
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(296, 1))
        self.conv2 = torch.nn.Conv2d(2, 4, kernel_size=(3, 1))
        # the input size of the first linear layer changes based on the feature extraction method used.
        if feature_method == "collective_features":
            self.linear1 = torch.nn.Linear(808, 50)
        else:
            self.linear1 = torch.nn.Linear(200, 50)

        # the output classification vector depends on the dataset
        if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
            self.linear2 = torch.nn.Linear(50, 7)
        elif dataset == "RAVDESS":
            self.linear2 = torch.nn.Linear(50, 8)
        else:
            self.linear2 = torch.nn.Linear(50, 5)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.15)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        x_out = self.linear1(x)
        x = F.relu(x_out)
        x = self.linear2(x)
        return x, x_out


'''
NN audio emotion recognition model
'''
class simple_NN(torch.nn.Module):
    def __init__(self, feature_method, dataset):
        super(simple_NN, self).__init__()
        # the input size of the first linear layer changes based on the feature extraction method used.
        if feature_method == "collective_features":
            self.linear1 = torch.nn.Linear(312, 512)
        else:
            self.linear1 = torch.nn.Linear(50, 512)
        self.linear2 = torch.nn.Linear(512, 50)
        # the output classification vector depends on the dataset
        if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
            self.linear3 = torch.nn.Linear(50, 7)
        elif dataset == "RAVDESS":
            self.linear3 = torch.nn.Linear(50, 8)
        else:
            self.linear3 = torch.nn.Linear(50, 5)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.15)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x_out = self.linear2(x)
        x = self.relu(x_out)
        x = self.linear3(x)
        return x, x_out


'''
LSTM text multimodal and textual emotion recognition model
'''
class LSTM_text_emotions(torch.nn.Module):
    def __init__(self, dataset, text_features, text_only, use_audio_model, audio_features):
        super(LSTM_text_emotions, self).__init__()
        self.n_hidden = 150
        self.exclude_audio = text_only
        # depending on the text feature extraction method chosen
        if text_features == "w2v":
            self.embedding_dim = 300
        else:
            self.embedding_dim = 768
        self.max_sen_len = 30
        # determines if the audio feature vector will be appended, thus changing the input size of the second LSTM layer.
        if self.exclude_audio:
            self.combined_input = 2 * self.n_hidden
        else:
            if use_audio_model:
                self.combined_input = 2 * self.n_hidden + 50
            else:
                if audio_features == "mfcc_only":
                    self.combined_input = 2 * self.n_hidden + 50
                else:
                    self.combined_input = 2 * self.n_hidden + 312

        # the number of classification classes of the used dataset
        if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
            self.n_class = 7
        else:
            self.n_class = 5
        self.droupout = torch.nn.Dropout(0.3)
        self.word_bilstm = torch.nn.LSTM(self.embedding_dim, self.n_hidden, batch_first=True, bidirectional=True)
        self.pos_bilstm = torch.nn.LSTM(self.combined_input, self.n_hidden, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(2 * self.n_hidden, self.n_class)
        self.attention = Attention(self.n_hidden, self.max_sen_len)

    def forward(self, x, audio):
        x, hidden_states = self.word_bilstm(x.float())
        x = self.droupout(x)
        s = self.attention(x)
        # if the model is meant to be used multimodally or only as text emotion classificator
        if self.exclude_audio:
            x_context, hidden_states = self.pos_bilstm(s.float())
        else:
            x_context, hidden_states = self.pos_bilstm(torch.cat([s, audio], -1).float())
        x = x_context.reshape(-1, 2 * self.n_hidden)
        pred_pos = F.softmax(self.linear(x), dim=-1)
        return pred_pos


'''
BERT multimodal or textual model
'''
class CustomBert(torch.nn.Module):
    def __init__(self, dataset, text_only, use_audio_model, audio_features):
        super(CustomBert, self).__init__()
        self.text_only = text_only
        self.config = AutoConfig.from_pretrained("bert-base-uncased")
        self.transformer = AutoModel.from_pretrained("bert-base-uncased", config=self.config)
        num_hidden_size = self.transformer.config.hidden_size
        # the number of classification classes of the used dataset
        if dataset == "ECF" or dataset == "ECF_FT2D" or dataset == "ECF_REPETSIM":
            self.n_class = 7
        else:
            self.n_class = 5
        # determines if the audio feature vector will be appended, thus changing the input size of the second LSTM layer.
        if text_only:
            self.linear_input_size = num_hidden_size
        else:
            if use_audio_model:
                self.linear_input_size = num_hidden_size + 50
            else:
                if audio_features == "mfcc_only":
                    self.linear_input_size = num_hidden_size + 50
                else:
                    self.linear_input_size = num_hidden_size + 312
        self.classifier = torch.nn.Linear(self.linear_input_size, self.n_class)

    def forward(self, input_ids, attention_masks, audio=None):
        hidden_states = self.transformer(input_ids=input_ids, attention_mask=attention_masks)
        concat = hidden_states.last_hidden_state[:, 0, :]
        # if the audio feature vector should be concatenated
        if not self.text_only:
            concat = torch.cat((concat, audio), dim=-1)
        concat = concat.float()
        output = self.classifier(concat)
        return output


'''
The Attention layer of the LSTM multimodal and textual model.
'''
class Attention(torch.nn.Module):
    def __init__(self, n_hidden, sen_len):
        super(Attention, self).__init__()
        self.n_hidden = n_hidden
        self.sen_len = sen_len
        self.linear1 = torch.nn.Linear(n_hidden * 2, n_hidden * 2)
        self.linear2 = torch.nn.Linear(n_hidden * 2, 1)

    def forward(self, x):
        x_tmp = x.reshape(-1, self.n_hidden * 2)
        u = torch.tanh(self.linear1(x_tmp))
        alpha = self.linear2(u)
        alpha = F.softmax(alpha.reshape(-1, 1, self.sen_len), dim=-1)
        x = torch.matmul(alpha, x).reshape(-1, self.n_hidden * 2)
        return x
