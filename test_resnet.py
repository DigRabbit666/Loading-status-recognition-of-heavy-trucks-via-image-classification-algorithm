import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST, ImageFolder, CIFAR10
import os
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models as models
import argparse
from my_residual_models import MyResnet


transform_test = transforms.Compose([
        transforms.Resize(size=(608, 608), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.48670617, 0.47717795, 0.47324306], [0.2036544, 0.20564124, 0.2122561]),
    ])

valset = ImageFolder(r'E:\zhonghuan_door\val',
                     transform=transform_test)
valloader = DataLoader(valset, batch_size=32, shuffle=True)
model = MyResnet(3)
model.eval()
model.cuda()
criterion = nn.CrossEntropyLoss()
model_path = 'model\\resnet18_{}.pth'.format('best')
assert os.path.isfile(model_path)
model.load_state_dict(torch.load(model_path))
validation_losses = []
validation_loss = 0
accuracy = 0
with torch.no_grad():
    for images, labels in tqdm(valloader):
        images = images.cuda()
        labels = labels.cuda()
        log_ps = model(images)[0]
        validation_loss += criterion(log_ps, labels)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
        validation_losses.append(validation_loss / len(valloader))
        cur_acc = accuracy / len(valloader)
    print("Test Loss: {:.3f}.. ".format(validation_loss / len(valloader)),
          "Test Accuracy: {:.3f}".format(accuracy / len(valloader)))
