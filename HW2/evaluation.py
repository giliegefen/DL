
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from torch.autograd import Variable

num_epochs = 4
learning_rate = 0.001
use_gpu = torch.cuda.is_available()



def to_cuda(x):
    if use_gpu:
        x = x.cuda()
    return x

# Model
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
        out = self.pr(out)
        out = self.fully(out)
        return self.logsoftmax(out)


def evaluate_hw2():
    cnn = CNN()
    cnn = to_cuda(cnn)

    test_loader = torch.utils.data.DataLoader(
        dsets.CIFAR10(root='./data/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.2434, 0.2615))
        ])))

    cnn.load_state_dict(torch.load('cnnMax400.pkl'))
    cnn.eval()
    
    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = to_cuda(images)
        labels = to_cuda(labels)
        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {:.4f}%'.format(100 * float(correct) / total))





if __name__ == "__main__":
    evaluate_hw2()
