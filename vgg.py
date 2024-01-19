from torch import nn
from torch.nn.init import xavier_uniform


class VGG(nn.Module):
    def __init__(self):
        super().__init__()

        # 3x224x224

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        xavier_uniform(self.conv1.weight)
        self.relu1 = nn.ReLU()
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # 64x224x224
        self.dropout1 = nn.Dropout(p=0.4)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 64x112x112

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        # 128x112x112

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 128x56x56

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()
        # 256x56x56
        
        self.dropout2 = nn.Dropout(p=0.4)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 256x56x56

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU()
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu10 = nn.ReLU()
        # 512x28x28

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 512x14x14

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu11 = nn.ReLU()
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu12 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu13 = nn.ReLU()
        # 512x14x14

        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.linear_shape = 512 * 7 * 7
        self.fc1 = nn.Linear(self.linear_shape, 4096)
        self.relu14 = nn.ReLU()

        self.fc2 = nn.Linear(4096, 4096)
        self.relu15 = nn.ReLU()

        self.fc3 = nn.Linear(4096, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv1_1(x)
        x = self.relu2(x)
        # print(f"1 shape: {x.shape}")
        
        x = self.dropout1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu3(x)
        x = self.conv2_1(x)
        x = self.relu4(x)

        # print(f"2 shape: {x.shape}")

        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu5(x)
        x = self.conv3_1(x)
        x = self.relu6(x)
        x = self.conv3_2(x)
        x = self.relu7(x)

        # print(f"3 shape: {x.shape}")
        
        x = self.dropout2(x)

        x = self.pool3(x)

        # print(f"before pool 4 shape: {x.shape}")

        x = self.conv4(x)
        x = self.relu8(x)
        x = self.conv4_1(x)
        x = self.relu9(x)
        x = self.conv4_2(x)
        x = self.relu10(x)

        # print(f"4 shape: {x.shape}")

        x = self.pool4(x)

        x = self.conv5(x)
        x = self.relu11(x)
        x = self.conv5_1(x)
        x = self.relu12(x)
        x = self.conv5_2(x)
        x = self.relu13(x)

        # print(f"5 shape: {x.shape}")

        x = self.pool5(x)

        # print(f"6 shape: {x.shape}")

        x = x.view(-1, self.linear_shape)

        x = self.fc1(x)
        x = self.relu14(x)
        x = self.fc2(x)
        x = self.relu15(x)
        x = self.fc3(x)

        return x
