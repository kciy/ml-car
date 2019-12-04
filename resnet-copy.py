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
import datetime
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

loss_list = []
iteration_list = []
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
since = time.time()
best_acc = 0.0
step = 0
for epoch in range(num_epochs):
    for i, batch in enumerate(trainloader):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        ### TRAINING
        images, labels = batch["image"], batch["category"]
        train, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(train.type('torch.FloatTensor'))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        step += 1
        print(step)
        if step % 50 == 0:
            correct = 0
            total = 0
            ### VALIDATION
            model.eval()
            with torch.no_grad():
                for ii, batch_test in enumerate(valloader):
                    images, labels = batch_test["image"].to(device), batch_test["category"].to(device)
                    outputs = model.forward(images.type('torch.FloatTensor'))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0) # batch_size
                    correct += (predicted == labels).sum()
                    print('..........true:')
                    print(labels)
                    print('..........pred:')
                    print(predicted)
                    print('!!!!!!!!!!!!!!!!!!!!!!')
                    # accuracy_score = accuracy_score(labels, predicted)  # (tp + tn) / (tp + fp + tn + fn)
                    precision = precision_score(labels, predicted, labels=np.unique(predicted))  # tp / (tp + fp)
                    recall = recall_score(labels, predicted, labels=np.unique(predicted))  # tp / (tp + fn)
                    f1 = f1_score(labels, predicted, labels=np.unique(predicted))  # 2tp / (2tp + fp + fn)

                    # accuracy_list.append(accuracy_score)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    f1_list.append(f1)

                    accuracy = 1 * correct / float(total)

                    # print(accuracy_score)
                    print(accuracy)

                if accuracy > best_acc:
                    best_acc = accuracy

                wandb.log({"epoch": epoch, "loss": np.mean(loss_list), "accuracy mean": np.mean(accuracy_list), "precision": np.mean(precision_list), "recall": np.mean(recall_list), "f1": np.mean(f1_list)}, step=step)

                loss_list.append(loss.data)
                iteration_list.append(step)
                # accuracy_list.append(accuracy)
            # scheduler.step(loss)
            # print(f"Epoch {epoch+1}/{num_epochs}, It. {step}, Train loss: {loss.item()}, Validation accuracy: {accuracy}%")

            print(datetime.datetime.now(),
                  "Epoch: %d/%d" %(epoch+1, num_epochs),  
                  "TrainLoss: %.3f" %(loss.item()), 
                #   "TrainAccuracy: %.3f" %(training_accuracy), 
                #   "ValLoss: %.3f" %(sum_loss_validation/len(validation_dataframe)), 
                  "ValAccuracy: %.3f" %(accuracy),
                  "Best ValAccuracy: %.3f" %(best_acc))
            print('------------')
            model.train()
time_elapsed = time.time() - since
print('Training complete in {:0f}min {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val acc: {:4f}'.format(best_acc))
            
torch.save(model, 'ActiveCarModel2.pth')

plt.plot(loss_list, label='Train Loss')
plt.plot(accuracy_list, label='Validation Accuracy')
plt.grid()
plt.legend(frameon=False)
plt.show()
