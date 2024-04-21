import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel


class LSTM_extractor_w2v(torch.nn.Module):
    def __init__(self):
        super(LSTM_extractor_w2v, self).__init__()
        self.bilstm = torch.nn.LSTM(50000, 1024, batch_first=True, bidirectional=True)
        self.dp = torch.nn.Dropout(p=0.3)
        self.linear = torch.nn.Linear(2048, 512)
        self.linear2 = torch.nn.Linear(512, 7)

    def forward(self, x):
        x_states, hidden_states = self.bilstm(x)
        x = self.dp(x_states)
        x = self.linear(x)
        x = self.linear2(x)
        return x, x_states


class LSTM_extractor_old(torch.nn.Module):
    def __init__(self):
        super(LSTM_extractor_old, self).__init__()
        self.bilstm = torch.nn.LSTM(312, 400, batch_first=True, bidirectional=True)
        self.dp = torch.nn.Dropout(p=0.3)
        self.linear = torch.nn.Linear(800, 100)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(100, 7)

    def forward(self, x):
        x_states, hidden_states = self.bilstm(x)
        x = x_states.squeeze()
        x = self.dp(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = F.softmax(x)
        return x, x_states


class LSTM_extractor(torch.nn.Module):
    def __init__(self):
        super(LSTM_extractor, self).__init__()
        self.bilstm = torch.nn.LSTM(36, 300, batch_first=True, bidirectional=True)
        self.dp = torch.nn.Dropout(p=0.3)
        self.linear = torch.nn.Linear(600, 400)
        self.linear2 = torch.nn.Linear(400, 150)
        self.linear3 = torch.nn.Linear(150, 8)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x_states, hidden_states = self.bilstm(x)
        # x = self.dp(x_states)
        # out = x_states[:, -1]
        out = torch.cat((x_states[:, -1, 300:], x_states[:, 0, :300]), 1)
        x = F.relu(self.linear(out))
        x = self.dp(x)
        x = F.relu(self.linear2(x))
        x = self.dp(x)
        x = self.linear3(x)
        # x = F.softmax(x)
        return x, x_states


class ConvNet(torch.nn.Module):
    def __init__(self):
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
        self.fc1 = torch.nn.Linear(1280, 512)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(512, 100)
        self.fc3 = torch.nn.Linear(100, 7)

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


class CNN_small(torch.nn.Module):
    def __init__(self):
        super(CNN_small, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=(3, 1))
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(296, 1))
        self.conv2 = torch.nn.Conv2d(4, 8, kernel_size=(3, 1))
        self.linear1 = torch.nn.Linear(400, 100)
        self.linear2 = torch.nn.Linear(100, 7)
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


class simple_NN(torch.nn.Module):
    def __init__(self):
        super(simple_NN, self).__init__()
        self.linear1 = torch.nn.Linear(312, 840)
        self.linear2 = torch.nn.Linear(840, 100)
        self.linear3 = torch.nn.Linear(100, 7)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.15)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x_out = self.linear2(x)
        x = self.relu(x_out)
        x = self.linear3(x)
        return x, x_out


class LSTM_text_emotions(torch.nn.Module):
    def __init__(self):
        super(LSTM_text_emotions, self).__init__()
        self.n_hidden = 150
        self.embedding_dim = 300
        self.max_sen_len = 30
        self.doc_len = 41
        self.n_class = 7
        self.droupout = torch.nn.Dropout(0.3)
        self.word_bilstm = torch.nn.LSTM(self.embedding_dim, self.n_hidden, batch_first=True, bidirectional=True)
        self.pos_bilstm = torch.nn.LSTM(2 * self.n_hidden + 100, self.n_hidden, batch_first=True, bidirectional=True)
        # self.pos_bilstm = torch.nn.LSTM(2 * self.n_hidden, self.n_hidden, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(2 * self.n_hidden, self.n_class)
        self.attention = Attention(self.n_hidden, self.max_sen_len)

    def forward(self, x, audio):
        # x = x.reshape(-1, self.max_sen_len, self.embedding_dim)
        x, hidden_states = self.word_bilstm(x.float())
        x = self.droupout(x)
        s = self.attention(x)#.reshape(self.max_sen_len, 2 * self.n_hidden)
        # s = s.reshape(-1, self.doc_len, 2 * self.n_hidden)
        x_context, hidden_states = self.pos_bilstm(torch.cat([s, audio], -1).float())
        # x_context, hidden_states = self.pos_bilstm(s.float())
        x = x_context.reshape(-1, 2 * self.n_hidden)
        pred_pos = F.softmax(self.linear(x), dim=-1)
        # pred_pos = pred_pos.reshape(-1, self.doc_len, self.n_class)
        return pred_pos

class CustomBert(torch.nn.Module):
    def __init__(self):
        super(CustomBert, self).__init__()
        self.config = AutoConfig.from_pretrained("bert-base-uncased")
        self.transformer = AutoModel.from_pretrained("bert-base-uncased", config=self.config)
        num_hidden_size = self.transformer.config.hidden_size
        self.classifier = torch.nn.Linear(num_hidden_size + 25, 7)
        # self.classifier = torch.nn.Linear(num_hidden_size, 7)

    def forward(self, input_ids, attention_masks, audio=None):
        hidden_states = self.transformer(input_ids=input_ids, attention_mask=attention_masks)
        concat = hidden_states.last_hidden_state[:, 0, :]
        # audio = audio.squeeze()
        concat = torch.cat((concat, audio), dim=-1)
        concat = concat.float()
        output = self.classifier(concat)
        return output

class Attention(torch.nn.Module):
    def __init__(self, n_hidden, sen_len):
        super(Attention, self).__init__()
        self.n_hidden = n_hidden
        self.sen_len = sen_len
        self.linear1 = torch.nn.Linear(n_hidden*2, n_hidden*2)
        self.linear2 = torch.nn.Linear(n_hidden*2, 1)

    def forward(self, x):
        x_tmp = x.reshape(-1, self.n_hidden*2)
        u = torch.tanh(self.linear1(x_tmp))
        alpha = self.linear2(u)
        alpha = F.softmax(alpha.reshape(-1, 1, self.sen_len), dim = -1)
        x = torch.matmul(alpha, x).reshape(-1, self.n_hidden*2)
        return x

