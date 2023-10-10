import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=292, out_channels=64, kernel_size=3, padding=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=3)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=292, kernel_size=3, padding=3)
        self.conv4 = nn.Conv1d(in_channels=292, out_channels=512, kernel_size=3, padding=3)
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=1460, kernel_size=3, padding=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.RelU = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.RelU(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.RelU(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.RelU(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.RelU(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = self.RelU(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.RelU(x)
        x = self.fc2(x)
        x = self.RelU(x)
        x = self.fc3(x)
        x = self.RelU(x)
        return x