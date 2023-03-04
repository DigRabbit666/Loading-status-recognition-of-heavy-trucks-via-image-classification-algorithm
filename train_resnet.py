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
from my_residual_models import MyResnet


if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.Resize((608, 608), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4846409, 0.472469, 0.4687362], [0.20441233, 0.20865023, 0.2147461]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(608, 608), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.48670617, 0.47717795, 0.47324306], [0.2036544, 0.20564124, 0.2122561]),
    ])
    trainset = ImageFolder(r'E:\zhonghuan_door\train',
                           transform=transform_train)
    valset = ImageFolder(r'E:\zhonghuan_door\val',
                         transform=transform_test)
    trainloader = DataLoader(trainset, num_workers=2, batch_size=32, shuffle=True)
    valloader = DataLoader(valset, num_workers=2, batch_size=32, shuffle=True)
    model = MyResnet(class_num=3)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40, 50], gamma=0.1)
    epochs = 60
    train_losses, validation_losses = [], []
    best_acc = 0
    for e in range(epochs):
        running_loss = 0
        model.train()
        print('Current lr is: ', next(iter(optimizer.param_groups))['lr'])
        for images, labels in tqdm(trainloader):
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            output = model(images)[0]
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        validation_loss = 0
        accuracy = 0
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            for images, labels in tqdm(valloader):
                images = images.cuda()
                labels = labels.cuda()
                log_ps = model(images)[0]
                validation_loss += criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        train_losses.append(running_loss/len(trainloader))
        validation_losses.append(validation_loss/len(valloader))
        cur_acc = accuracy / len(valloader)
        if cur_acc > best_acc:
            save_path = 'model\\resnet18_{}.pth'.format('best')
            torch.save(model.state_dict(), save_path)
            best_acc = cur_acc
            print('Model is saved to {}'.format(save_path))
        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.3f}.. ".format(validation_loss/len(valloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(valloader)))


    plt.plot(train_losses, label='Training loss')
    plt.plot(validation_losses, label='Validation loss')
    plt.legend(frameon=False)
