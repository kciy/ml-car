import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from IRDataset import IRDataset
import sys
import time
# from time import datetime
from PIL import Image
import cv2
import wandb
# from visdom import Visdom
# from vis import Visualizations

data_dir = 'drive_all_day_train'

wandb.init(project="active-car-project")
wandb.config["more"] = "custom"

image_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.35675976, 0.37380189, 0.3764753], [0.32064945, 0.32098866, 0.32325324])
                                   ])
num_epochs = 10
batch_size = 32
valid_size = 0.1

# load data
def load_split_train_val(datadir, valid_size=0.1):
    train_data = IRDataset(data_dir, transform=image_transforms)
    val_data = IRDataset(data_dir, transform=image_transforms)
    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=batch_size)
    valloader = torch.utils.data.DataLoader(val_data,
                   sampler=val_sampler, batch_size=batch_size)

    return trainloader, valloader

dset = IRDataset(data_dir, transform=image_transforms)
print('Number of images: ', len(dset))
print('Number of categories: ', len(dset.categories))

trainloader, valloader = load_split_train_val(data_dir, valid_size)

images, labels = next(iter(trainloader))
model = torchvision.models.resnet18(pretrained=True)

# vis = Visualizations()
device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
# freeze
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
#for name, child in model.named_children():
#    print(name)

model.fc = nn.Sequential(nn.Linear(512, 512), # 2048, 512
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10),
                                 nn.LogSoftmax(dim=1))
                    
criterion = nn.NLLLoss().to(device)  # loss function (negative log-likelihood)

optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
# optimizer = optim.Adam(model.fc.parameters(), lr=1e-2)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
model.to(device)

# loss_list = [0]
# iteration_list = [0]
# accuracy_list = [0]
# precision_list = [0]
# recall_list = [0]
# f1_list = [0]
'''
since = time.time()
best_acc = 0.0
val_acc_history = []
step = 0
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            loader = trainloader
        else: 
            model.eval()
            loader = valloader

        running_loss = 0.0
        running_corrects = 0

        for i, batch in enumerate(loader):
            inputs = batch["image"].to(device)
            labels = batch["category"].to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                if phase == 'train':
                    outputs = model(inputs.type('torch.FloatTensor'))
                    loss = criterion(outputs, labels)
                preds = torch.argmax(outputs.data, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    step += 1

            print(step)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum() # (predicted == labels).sum()

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = running_corrects.double() / len(loader.dataset)

        print("{} loss: {:.4f}......... {} acc: {:.4f}".format(phase, epoch_loss, phase, epoch_acc))

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
        if phase == 'val':
            val_acc_history.append(epoch_acc)
    
    print()

time_elapsed = time.time() - since
print('Training complete in {:0f}min {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val acc: {:4f}'.format(best_acc))
'''