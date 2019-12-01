import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim, nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models import resnet50
import projutils

device = torch.device('cuda:0')

if __name__ == "__main__":
    label_file = 'data/train_labels.csv'
    train_path = 'data/train/'

    model = resnet50(num_classes=2).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    t = transforms.Compose([projutils.ToTensor()])
    train, test = projutils.train_test_loader(
        label_file, train_path, transform=t, train_size=0.8)

    loss_value = np.zeros(1000)
    test_acc = np.zeros(1000)
    train_acc = np.zeros(1000)
    for epoch in range(1000):
        running_loss = 0.0
        total_loss = 0
        train_correct = 0
        test_correct = 0
        train_total = 0
        test_total = 0
        for i, data in enumerate(train):
            inputs, labels = data['image'], data['label']
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()
            train_total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        for i, data in enumerate(test):
            inputs, labels = data['image'], data['label']
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = Variable(inputs), Variable(labels)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

        loss_value[epoch] = total_loss
        test_acc[epoch] = test_correct / test_total
        train_acc[epoch] = train_correct / train_total

    plt.plot(np.arange(1000), loss_value)
    plt.plot(np.arange(1000), test_acc)
    plt.plot(np.arange(1000), train_acc)
    plt.show()
