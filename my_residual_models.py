import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
import torchvision


class MyResnet(nn.Module):
    def __init__(self, class_num=10):
        super(MyResnet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 128)
        self.classifier = nn.Linear(128, class_num)

    def forward(self, x):
        fm = self.model(x)
        x = self.classifier(fm)
        return x, fm


class MyResnetEncoder(nn.Module):
    def __init__(self):
        super(MyResnetEncoder, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 128)

    def forward(self, x):
        fm = self.model(x)
        return fm


def test():
    net = MyResnet()
    print(net)
    y, fm = net(torch.randn(1, 3, 224, 224))
    print(y.size())
    print(fm.size())


def test1():
    dummy_input = Variable(torch.randn(10, 3, 224, 224))
    model = torchvision.models.resnet18(pretrained=True)
    y = model(dummy_input)
    print(y.size())


if __name__ == '__main__':
    test()


