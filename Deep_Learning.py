from data import load_data
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

from torch.autograd import Variable

device = torch.device('cuda:0')
label_file = 'data/train_labels.csv'
train_path = 'data/train/'


class simpleCNN(nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 21 * 21, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        print(x.size())
        x = self.pool(F.relu(self.conv1(x)))
        print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        print(x.size())
        x = x.view(-1, 16 * 21 * 21)
        print(x.size())
        x = F.relu(self.fc1(x))
        print(x.size())
        x = F.relu(self.fc2(x))
        print(x.size())
        x = self.fc3(x)
        return x


model = simpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)


def train():

    data, y = load_data(label_file, train_path, num=1000)
    data = torch.from_numpy(data).float().to(device)
    y = torch.from_numpy(y).long().to(device)
    data = data.permute(0, 3, 1, 2)
    # train, test, y_train, y_test = train_test_split(data, y, train_size=0.8)
    # train, y_train = torch.from_numpy(train), torch.from_numpy(y_train)

    for epoch in range(2):
        running_loss = 0.0
        inputs, labels = Variable(data), Variable(y)
        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]


if __name__ == "__main__":
    print('__Python VERSION:', sys.version)
    print('__PyTorch VERSION:', torch.__version__)
    #print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    train()
