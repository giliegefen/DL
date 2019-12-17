######## full example - cifar10 ######

# load and normalize data, define hyper parameters:

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# Hyper Parameters
num_epochs = 400
learning_rate = 0.001

# Image Preprocessing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.2434, 0.2615)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.2434, 0.2615)),
])


# CIFAR-10 Dataset
train_dataset = dsets.CIFAR10(root='./data/',
                              train=True,
                              transform=transform_train,
                              download=True)

test_dataset = dsets.CIFAR10(root='./data/',
                             train=False,
                             transform=transform_test,
                             download=True)


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=80,
                                          shuffle=False,
                                          num_workers=2)

# define a model:
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=2),
            nn.BatchNorm2d(16),
            nn.PReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.PReLU(), nn.MaxPool2d(2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(), nn.MaxPool2d(2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),nn.MaxPool2d(2))

        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, 19)
        self.pr= nn.PReLU()
        self.fully = nn.Linear(19, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.drop(out)
        out = self.fc(out)
        out=self.pr(out)
        out=self.fully(out)
        return self.logsoftmax(out)

# Model

use_gpu = torch.cuda.is_available()  # global bool


def to_cuda(x):
    if use_gpu:
        x = x.cuda()
    return x


cnn = CNN()
cnn = to_cuda(cnn)

epochs_loss=[]
epochs_err=[]
trainloss=[]
testloss=[]
train_err=[]
test_err=[]

# convert all the weights tensors to cuda()
# Loss and Optimizer

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

print ('number of parameters: ', sum(param.numel() for param in cnn.parameters()))
# training the model
cnn.train()  # turning the network to training mode, affect dropout and batch-norm layers if exists
for epoch in range(num_epochs):
    total_tr = 0
    correct_tr = 0
    for i, (images, labels) in enumerate(train_loader):
        images = to_cuda(images)
        labels = to_cuda(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)
        total_tr += labels.size(0)
        correct_tr += (predicted == labels).sum()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


    cnn.eval()
    correct = 0
    total = 0
    test_loss = 0
    for images, labels in test_loader:
        images = to_cuda(images)
        labels = to_cuda(labels)

        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    loss_te = criterion(outputs, labels)
    testloss.append(float(loss_te.item()))
    test_err.append(1 - (float(correct) / total))

    epochs_loss.append(epoch + 1)
    epochs_err.append(epoch + 1)
    trainloss.append(float(loss.item()))
    train_err.append(1 - (float(correct_tr) / total_tr))


# evaluating the model
cnn.eval()  # turning the network to evaluation mode, affect dropout and batch-norm layers if exists
correct = 0
total = 0
test_loss=0
for images, labels in test_loader:
    images = to_cuda(images)
    labels = to_cuda(labels)
    outputs = cnn(images)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print ('Accuracy of the model on the 10000 test images: {:.4f}%'.format(100 * float(correct) / total))

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnnMax400.pkl')

def create_plot_loss(x,y):
    plt.figure(1)
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(x, y)
    plt.gcf().legend(('train', 'test'))
    plt.savefig('lossNew400.png')

def create_plot_err(x,y):
    plt.figure(2)
    plt.title("Model Error")
    plt.xlabel("Epochs")
    plt.plot(x, y)
    plt.gcf().legend(('train' , 'test'))
    plt.ylabel("Error")
    plt.savefig('errorNew400.png')


all_loss=[]
all_err=[]
for i in range(num_epochs):
    all_loss.append([trainloss[i],testloss[i]])
    all_err.append([train_err[i], test_err[i]])

create_plot_err(epochs_err,all_err)
create_plot_loss(epochs_loss,all_loss)