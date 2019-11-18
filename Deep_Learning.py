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
label_file = 'D:/pycharmProject/data/histopathologic-cancer-detection/train_labels.csv'
train_path = 'D:/pycharmProject/data/histopathologic-cancer-detection/train/'

class simpleCNN(nn.module):
    def __init__(self):
        super(simpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = simpleCNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

def train():
    ## TO DO: TRAIN
    for epoch in range(2):

        running_loss = 0.0
        data, y = load_data(label_file, train_path, num=5000)
        inputs, labels = Variable(data), Variable(y)
        optimizer.zero_grad()

if __name__ == "__main__":
    print('__Python VERSION:', sys.version)
    print('__PyTorch VERSION:', torch.__version__)
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())