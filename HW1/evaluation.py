
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg') # allows it to operate on machines without graphical interface
import matplotlib.pyplot as plt
matplotlib.use('Agg') # allows it to operate on machines without graphical interface
# import matplotlib.pyplot as plt

input_size = 784
hidden_size = 76
num_classes = 10
num_epochs = 120
batch_size = 35
learning_rate = 0.1
use_gpu = torch.cuda.is_available()

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
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        #self.log = nn.LogSigmoid()
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


def evaluate_hw1():
    net = NeuralNet(input_size, num_classes)
    net = to_cuda(net)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])))
    net.load_state_dict(torch.load('model.pkl'))


    # Test the Model
    correct = 0
    total = 0
    #net.eval()  # turning the network to evaluation mode, affect dropout and batch-norm layers if exists
    for images, labels in test_loader:
        images = to_cuda(images.view(-1, 28 * 28))
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

    print('Accuracy of the model on the 10000 test images: {:.4f}%'.format(100 * float(correct) / total))


if __name__ == "__main__":
    evaluate_hw1()
