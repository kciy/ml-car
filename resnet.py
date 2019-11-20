import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from IRDataset import IRDataset
import sys
from PIL import Image
import cv2
from visdom import Visdom
from vis import Visualizations

data_dir = 'drive_day_2019_08_21_16_14_06/test_combined' # 1263 images

image_transforms = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.35675976, 0.37380189, 0.3764753], [0.32064945, 0.32098866, 0.32325324])
                                   ])

# load data
def load_split_train_test(datadir, valid_size=0.1):
    train_data = IRDataset(data_dir, transform=image_transforms)
    test_data = IRDataset(data_dir, transform=image_transforms)
    
    train_data_old = datasets.ImageFolder(datadir, transform=image_transforms)
    test_data_old = datasets.ImageFolder(datadir, transform=image_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=10)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=10)

    return trainloader, testloader

dset = IRDataset(data_dir, transform=image_transforms)
print('Number of images: ', len(dset))
print('Number of categories: ', len(dset.categories))

trainloader, testloader = load_split_train_test(data_dir, 0.2)

images, labels = next(iter(trainloader))
model = torchvision.models.resnet18(pretrained=True)

vis = Visualizations()
device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
# freeze
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
for name, child in model.named_children():
    print(name)

model.fc = nn.Sequential(nn.Linear(512, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10),
                                 nn.LogSoftmax(dim=1))
                    
error = nn.NLLLoss()  # negative log-likelihood
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
model.to(device)

num_epochs = 20
batch_size = 1

loss_list = []
iteration_list = []
accuracy_list = []
step = 0
for epoch in range(num_epochs):
    for i, batch in enumerate(trainloader):
        ### TRAINING
        images, labels = batch["image"], batch["category"]
        train, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(train)
        loss = error(outputs, labels)
        loss.backward()
        optimizer.step()
        step += 1
        if step % 50 == 0:
            correct = 0
            total = 0
            ### VALIDATION
            model.eval()
            with torch.no_grad():
                for ii, batch_test in enumerate(testloader):
                    images, labels = batch_test["image"], batch_test["category"]
                    images, labels = images.to(device), labels.to(device)
                    outputs = model.forward(images)
                    predicted = torch.argmax(outputs.data, 1)
                    print(f"predicted (argmax) {predicted} ?= {labels} labels")
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                accuracy = 100 * correct / float(total)
                # print(f"accuracy ({accuracy}) = 100 * correct ({correct}) / total ({float(total)})")

                vis.plot_loss(np.mean(loss_list), step)
                vis.plot_acc(accuracy, step)

                loss_list.append(loss.data)
                iteration_list.append(step)
                accuracy_list.append(accuracy)
                
            print(f"Epoch {epoch+1}/{num_epochs}, It. {step}, Train error: {loss.item()}, Test accuracy: {accuracy}%")
            model.train()

torch.save(model, 'ActiveCarModel1.pth')

plt.plot(loss_list, label='Train Loss')
plt.plot(accuracy_list, label='Test Accuracy')
plt.grid()
plt.legend(frameon=False)
plt.show()
