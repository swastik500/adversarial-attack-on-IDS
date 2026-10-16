import torch
import torch.nn as nn
from torch.nn import functional as F

class IDS_model(nn.Module):
	def __init__(self):
		super(IDS_model, self).__init__()
		self.sequence = nn.Sequential(
			nn.Linear(78,128),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(128,64),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(64,32),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(32,1),
			nn.Sigmoid()
			)

	def forward(self, x):
		output = self.sequence(x)

		return output

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        H1 = 40
        H2 = 40
        H3 = 32
        H4 = 16
        H5 = 8

        # Four Hidden Layers
        self.fc1 = nn.Linear(78, H1)
        self.fc1_1 = nn.Linear(H1, H2)
        self.fc1_2 = nn.Linear(H2, H3)
        self.fc1_3 = nn.Linear(H3, H4)
        self.fc21 = nn.Linear(H4, H5)
        self.fc22 = nn.Linear(H4, H5)
        self.fc3 = nn.Linear(H5, H4)
        self.fc3_1 = nn.Linear(H4, H3)
        self.fc3_2 = nn.Linear(H3, H2)
        self.fc3_3 = nn.Linear(H2, H1)
        self.fc4 = nn.Linear(H1, 78)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc1_1(h1))
        h1 = F.relu(self.fc1_2(h1))
        h1 = F.relu(self.fc1_3(h1))
        
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = F.relu(self.fc3_1(h3))
        h3 = F.relu(self.fc3_2(h3))
        h3 = F.relu(self.fc3_3(h3))

        return torch.sigmoid(self.fc4(h3))
        # return self.fc4(h3)
        # return torch.tanh(self.fc4(h3))
        # return F.relu(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 78))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class IDS_CNN(nn.Module):
    def __init__(self):
        super(IDS_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.fc1 = nn.Linear(32 * 19, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

class IDS_RNN(nn.Module):
    def __init__(self):
        super(IDS_RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(2)
        h0 = torch.zeros(2, x.size(0), 32).to(x.device)
        c0 = torch.zeros(2, x.size(0), 32).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.sigmoid(self.fc(out))
        return out