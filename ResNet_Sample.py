import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim, nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models import resnet50
import projutils

device = torch.device('cuda:0')


def train(label_file, train_path, load_file='', num_step=10):
    '''Train the neural networks

    Args:
        label_file (str): Path to the csv file that has label information
        train_path (str): Path to the training dataset folder
        load_file (str, optional): State file that is needed to be loaded to the
            model
        num_step (int. optional): Number of epochs to run

    Returns:
        num_step (int): Number of epochs runned
        loss_value (np.ndarray): Information of loss value
        train_acc (np.ndarray): Accuracy of each epoch on train set
        test_acc (np.ndarray): Accuracy of each epoch on test set

    '''
    model = resnet50(num_classes=2).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    init_epoch = 0

    if load_file:
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
        criterion.load_state_dict(checkpoint['loss'])

    t = transforms.Compose([projutils.ToTensor()])
    train, test = projutils.train_test_loader(
        label_file, train_path, transform=t, train_size=0.8)

    loss_value = np.zeros(num_step)
    test_acc = np.zeros(num_step)
    train_acc = np.zeros(num_step)

    for epoch in range(10):
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
        torch.save({'epoch': epoch+init_epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion.state_dict()}, load_file)

    return num_step, loss_value, train_acc, test_acc


if __name__ == "__main__":
    label_file = 'data/train_labels.csv'
    train_path = 'data/train/'
    save_file = 'state.pt'

    steps, loss_value, train_acc, test_acc = train(
        label_file, train_path, load_file=save_file)

    plt.plot(np.arange(steps), loss_value)
    plt.plot(np.arange(steps), test_acc)
    plt.plot(np.arange(steps), train_acc)
    plt.show()
