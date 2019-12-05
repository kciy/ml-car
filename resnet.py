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
import time
import datetime
import wandb

data_dir = 'drive_all_day_train'

wandb.init(project="active-car-project")
wandb.config["more"] = "custom"

image_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5094, 0.5094, 0.5094], [0.3474, 0.3474, 0.3474])
                                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])

print(f'Starting time: {datetime.datetime.now()}')

num_epochs = 100
batch_size = 64
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
    train_loader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data,
                   sampler=val_sampler, batch_size=batch_size)

    return train_loader, val_loader

dset = IRDataset(data_dir, transform=image_transforms)
print('Number of images: ', len(dset))
print('Number of categories: ', len(dset.categories))

train_loader, val_loader = load_split_train_val(data_dir, valid_size)
images, labels = next(iter(train_loader))
model = torchvision.models.resnet18(pretrained=True)

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
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
model.to(device)

val_acc_history = []
val_loss_history = []
train_acc_history = []
train_loss_history = []

precision_history = []
recall_history = []
f1_history = []

def validate(model):
    val_acc = 0.0
    val_loss = 0.0
    model.eval()
    running_precision = []
    running_recall = []
    running_f1 = []
    with torch.no_grad():
        ii = 0
        for i, batch in enumerate(val_loader):
            ii += 1
            print(ii, end='\r')
            images, labels = batch["image"].to(device), batch["category"].to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_acc += torch.sum(labels.data == preds).item()
            
            precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy())
            recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy())
            f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy())

            running_precision.append(precision)
            running_recall.append(recall)
            running_f1.append(f1)
            
        precision_history.append(np.mean(running_precision))
        recall_history.append(np.mean(running_recall))
        f1_history.append(np.mean(running_f1))

        return val_loss, val_acc

def train(num_epochs):
    best_acc = 0.0
    step = 0
    for epoch in range(num_epochs):
        model.train()
        torch.set_grad_enabled(True)
        since = time.time()
        train_loss = 0.0
        train_acc = 0.0
        ii = 0
        for i, batch in enumerate(train_loader):
            ii += 1
            print(ii, end='\r')
            images, labels = batch["image"].to(device), batch["category"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            step += 1
            
            train_loss += loss.item() * images.size(0)
            train_acc += torch.sum(preds == labels.data).item() 
        
        print("\n")
        val_loss, val_acc = validate(model)
        
        scheduler.step()

        # 10% validation 90% training set
        len_train_loader = 2698 
        len_val_loader = 299

        train_acc /= len_train_loader
        train_loss /= len_train_loader

        val_acc /= len_val_loader
        val_loss /= len_val_loader

        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)
        
        wandb.log({"train loss": train_loss, "val loss": val_loss, "train acc": train_acc, "val acc": val_acc}, step=epoch)

        time_elapsed = time.time() - since
        print("Duration: {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60),
              "Epoch {}/{}".format(epoch+1, num_epochs),
              "Train Accuracy: {:.3f}".format(train_acc),
              "Validation Accuracy: {:.3f}".format(val_acc),
              "Train Loss: {:.3f}".format(train_loss),
              "Validation Loss: {:.3f}".format(val_loss),
             )

def plot_metrics():
    plt.figure(1)
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.grid()
    plt.legend(frameon=False)

    plt.figure(2)
    plt.plot(val_loss_history, label="Validation Loss")
    plt.plot(train_loss_history, label="Train Loss")
    plt.grid()
    plt.legend(frameon=False)

    plt.figure(3)
    plt.plot(precision_history, label="Precision")
    plt.grid()
    plt.legend(frameon=False)
    
    plt.figure(4)
    plt.plot(recall_history, label="Recall")
    plt.grid()
    plt.legend(frameon=False)
    
    plt.figure(5)
    plt.plot(f1_history, label="F1 Score")
    plt.grid()
    plt.legend(frameon=False)

    plt.show()


if __name__ == "__main__":
    since = time.time()

    train(num_epochs)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    torch.save(model, 'ActiveCarModel6.pth')
    plot_metrics()
