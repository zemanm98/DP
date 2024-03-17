import torch
from torch.nn import functional as F

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
        self.bilstm = torch.nn.LSTM(312, 512, batch_first=True, bidirectional=True)
        self.dp = torch.nn.Dropout(p=0.3)
        self.linear = torch.nn.Linear(1024, 512)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(512, 7)

    def forward(self, x):
        x_states, hidden_states = self.bilstm(x)
        x = self.dp(x_states)
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
        self.dropout = torch.nn.Dropout(0.2)
        self.conv4 = torch.nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2)
        self.pool4 = torch.nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(704, 32)
        self.dropout2 = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(32, 8)

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
        x = self.fc2(x)
        return F.softmax(x, dim=1), []

class CNN_small(torch.nn.Module):
    def __init__(self):
        super(CNN_small, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=(1, 4), stride=(1, 2))
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 2))
        self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=(1, 4), stride=(1, 2))
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(5, 2))



    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        return x

