from torch import nn


class VGGS(nn.Module):
    def __init__(self):
        super().__init__()

        # 3x112x112

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 64x56x56

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 128x28x28

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 256x14x14
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 512x7x7

        self.linear_shape = 512 * 7 * 7
        self.fc1 = nn.Linear(self.linear_shape, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)

    def forward(self, x):
        # print(f"input shape: {x.shape}")
        x = self.conv1(x)
        x = self.pool1(x)

        # print(f"1 shape: {x.shape}")

        x = self.conv2(x)
        x = self.pool2(x)

        # print(f"2 shape: {x.shape}")

        x = self.conv3(x)
        x = self.conv3_1(x)
        x = self.pool3(x)

        # print(f"3 shape: {x.shape}")

        x = self.conv4(x)
        x = self.conv4_1(x)
        x = self.pool4(x)

        # print(f"4 shape: {x.shape}")

        x = x.view(-1, self.linear_shape)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
