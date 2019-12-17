import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Hyper Parameters
input_size = 784
hidden_size = 76
num_classes = 10
num_epochs = 80
batch_size = 35
learning_rate = 0.1
use_gpu = torch.cuda.is_available()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3015,))])

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transform,
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transform)

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
# "taste" the data
it = iter(train_loader)
im, _ = it.next()
torchvision.utils.save_image(im, './data/example.png')


def to_cuda(x):
    if use_gpu:
        x = x.cuda()
    return x


# Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        #self.dropout = nn.Dropout2d(p=0.25)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        #self.log = nn.LogSoftmax()
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


net = NeuralNet(input_size, num_classes)
net = to_cuda(net)
epochs=[]
trainloss=[]
testloss=[]
train_err=[]
test_err=[]


# Loss and Optimizer
# Softmax is internally computed.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
#optimizer = torch.optim.RMSprop(net.parameters(),learning_rate,0.9)


print('number of parameters: ', sum(param.numel() for param in net.parameters()))
# Training the Model
net.train()  # turning the network to training mode, affect dropout and batch-norm layers if exists

for epoch in range(num_epochs):

    total_tr=0
    correct_tr=0
    for i, (images, labels) in enumerate(train_loader):
        images = to_cuda(images.view(-1, 28 * 28))
        labels = to_cuda(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total_tr += labels.size(0)
        correct_tr += (predicted.cpu() == labels).sum()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    correct = 0
    total = 0
    net.eval()  # turning the network to evaluation mode, affect dropout and batch-norm layers if exists
    test_loss = 0


    for images, labels in test_loader:
        images = to_cuda(images.view(-1, 28 * 28))
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    loss_te = criterion(outputs, labels)
    testloss.append(float(loss_te.item()))
    test_err.append(1 - (float(correct) / total))
    print('Accuracy of the model on the 10000 test images: {:.4f}%'.format(100 * float(correct) / total))
    print('Accuracy of the model on the train images: {:.4f}%'.format(100 * float(correct_tr) / total_tr))

    print('Epoch: [%d/%d], Step: [%d/%d], Loss_train: %.6f'
          % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))
    print('Epoch: [%d/%d], Step: [%d/%d], Loss_test: %.6f'
          % (epoch + 1, num_epochs, i + 1, len(test_dataset) // batch_size, loss_te.item()))
    epochs.append(epoch + 1)
    trainloss.append(float(loss.item()))
    train_err.append(1 - (float(correct_tr) / total_tr))


# Test the Model
correct = 0
total = 0
net.eval()  # turning the network to evaluation mode, affect dropout and batch-norm layers if exists
test_loss=0

for images, labels in test_loader:
    images = to_cuda(images.view(-1, 28 * 28))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
print('Accuracy of the model on the 10000 test images: {:.4f}%'.format(100 * float(correct) / total))

# Save the Model
torch.save(net.state_dict(), 'model.pkl')

plt.xlabel("Epochs")
y=[]
for i in range(num_epochs):
    y.append([trainloss[i],testloss[i]])
plt.plot(epochs,y)
plt.ylabel("Loss")
plt.show()

plt.xlabel("Epochs")
y=[]
for i in range(num_epochs):
    y.append([train_err[i],test_err[i]])
plt.plot(epochs,y)
plt.ylabel("Error")
plt.show()

