import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import AlexNet
from sklearn.model_selection import train_test_split
from data import load_data

if __name__ == "__main__":
    data, y = load_data('data/train_labels.csv', 'data/train/', num=10000)

    train, test, y_train, y_test = train_test_split(data, y, train_size=0.8)

    model = AlexNet(num_classes=2)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train, y_train = torch.from_numpy(train), torch.from_numpy(y_train)
    train, y_train = Variable(train), Variable(y_train)

    optimizer.zero_grad()

    out = model(train)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()
