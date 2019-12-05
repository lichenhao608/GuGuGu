import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim, nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.tensorboard import SummaryWriter
import projutils

device = torch.device('cuda:0')


def train(label_file, train_path, tensorboard_file='tensorboard', load_file='', num_step=10):
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
    tensorb = SummaryWriter(log_dir=tensorboard_file)

    if load_file:
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
        tensorb = checkpoint['loss']

    t = transforms.Compose([projutils.ToTensor()])
    train, test = projutils.train_test_loader(
        label_file, train_path, seed=0, transform=t, train_size=0.8)

    for epoch in range(num_step):
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
                tensorb.add_scalar(
                    'Loss/single_loss', running_loss, (epoch+init_epoch)*22+i, epoch+init_epoch)
                running_loss = 0.0

        for i, data in enumerate(test):
            inputs, labels = data['image'], data['label']
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = Variable(inputs), Variable(labels)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

        tensorb.add_scalar('Loss', total_loss, init_epoch+epoch)
        tensorb.add_scalar('Accuracy/test',
                           test_correct / test_total,
                           init_epoch + epoch
                           )
        tensorb.add_scalar('Accuracy/train',
                           train_correct / train_total,
                           init_epoch + epoch
                           )

        torch.save({'epoch': epoch+init_epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, load_file)

    tensorb.closs()


if __name__ == "__main__":
    label_file = 'data/train_labels.csv'
    train_path = 'data/train/'
    load_file = 'state.pt'

    steps, loss_value, train_acc, test_acc = train(
        label_file, train_path, load_file=load_file)
