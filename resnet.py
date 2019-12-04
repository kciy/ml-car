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
from sklearn.metrics import confusion_matrix
from IRDataset import IRDataset
import sys
import time
import datetime
from PIL import Image
import cv2
import wandb

data_dir = 'drive_all_day_train'

wandb.init(project="active-car-project")
wandb.config["more"] = "custom"

image_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.35675976, 0.37380189, 0.3764753], [0.32064945, 0.32098866, 0.32325324])
                                   ])
num_epochs = 10
print_every = 50
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
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(512, 512), # 2048, 512
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10),
                                 nn.LogSoftmax(dim=1))
                    
criterion = nn.NLLLoss().to(device)  # loss function (negative log-likelihood)

optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9) # optimizer = optim.Adam(model.fc.parameters(), lr=1e-2)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
model.to(device)

global val_acc_history
global precision_history
global recall_history
global f1_history
global best_acc
val_acc_history = []
precision_history = []
recall_history = []
f1_history = []

def validate(model):
    global val_acc_history
    global precision_history
    global recall_history
    global f1_history
    val_acc = 0.0
    val_loss = 0.0
    for i, batch in enumerate(valloader):
        images, labels = batch["image"].to(device), batch["category"].to(device)
        outputs = model(images.type('torch.FloatTensor'))
        val_loss += criterion(outputs, labels).item()

        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1])
        val_acc += equality.type(torch.FloatTensor).mean()
        '''
        _, prediction = torch.max(outputs.data, 1)

        tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        precision_history.append(precision)
        recall_history.append(recall)
        f1_history.append(f1)

        val_acc += torch.sum(prediction == labels.data)
        val_acc = val_acc.item() / 32
        val_acc_history.append(val_acc)
        '''

        return val_loss, val_acc

def train(num_epochs, print_every):
    best_acc = 0.0
    step = 0
    for epoch in range(num_epochs):
        model.train()
        since = time.time()
        train_acc = 0.0
        train_loss = 0.0

        for i, batch in enumerate(trainloader):
            images, labels = batch["image"].to(device), batch["category"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            step += 1
            step_percent = 100 * step / 85 % 100
            print(" {:.2f}%".format(step_percent), end='\r')
            
            train_loss += loss.item() # * images.size(0)

            if step % print_every == 0:
                model.eval()
                with torch.no_grad():
                    val_loss, val_acc = validate(model)
            
                train_loss = train_loss / len(trainloader.dataset)

        # val_acc = validate()

                if val_acc > best_acc:
                    print('got better')
                    best_acc = val_acc
                print(f"Best accuracy so far: {best_acc}")

                time_elapsed = time.time() - since
                print("Duration: {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60),
                    "Epoch {}/{}".format(epoch+1, num_epochs),
                    "Validation Accuracy: {:.3f}".format(val_acc),
                    "Train Loss: {:.3f}".format(train_loss),
                    "Validation Loss: {:.3f}".format(val_loss),
                    )
                
                train_loss = 0


if __name__ == "__main__":
    since = time.time()
    train(num_epochs, print_every)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))

    torch.save(model, 'ActiveCarModel4.pth')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.grid()
    plt.legend(frameon=False)
    plt.show()
