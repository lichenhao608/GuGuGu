import torch
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
        label_file, train_path, transform=t, train_size=0.1)

    for epoch in range(2):
        running_loss = 0.0
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
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
