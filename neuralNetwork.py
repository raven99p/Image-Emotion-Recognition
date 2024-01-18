from torch import nn
from torch.nn.init import xavier_uniform


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        xavier_uniform(self.conv1.weight)
        self.batch_norm1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.4)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.linear_shape = 64 * 56 * 56
        self.fc1 = nn.Linear(self.linear_shape, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = x.view(-1, self.linear_shape)

        x = self.fc1(x)
        x = self.fc2(x)

        return x
