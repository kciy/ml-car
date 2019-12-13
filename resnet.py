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

dir_0 = 'drive_all_night_train'
dir_1 = 'drive_all_day_train'
dir_2 = 'drive_all_train'

#####
data_dir = dir_1
modelname = 'ActiveCarModel21.pth'
#####

mean_0 = [0.6968, 0.6968, 0.6968]
std_0 = [0.2580, 0.2580, 0.2580]

mean_1 = [0.5094, 0.5094, 0.5094]
std_1 = [0.3474, 0.3474, 0.3474]

mean_2 = [0.5432, 0.5432, 0.5432]
std_2 = [0.3317, 0.3317, 0.3317]

if data_dir == dir_0:
    mean, std = mean_0, std_0
if data_dir == dir_1:
    mean, std = mean_1, std_1
if data_dir == dir_2:
    mean, std = mean_2, std_2

wandb.init(project="active-car-project")
wandb.config["more"] = "custom"

image_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std),
                                   ])

print(f'Starting time: {datetime.datetime.now()}')

num_epochs = 50
batch_size = 64
valid_size = 0.1

# load data
def load_split_train_val(datadir, size=0.1):
    train_data = IRDataset(data_dir, transform=image_transforms)
    val_data = IRDataset(data_dir, transform=image_transforms)
    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(size * num_train))
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
images, labels, _ = next(iter(train_loader))
model = torchvision.models.resnet18(pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(512, 512), # 2048, 512
                                 nn.LeakyReLU(0.3), 
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss().to(device)  # negative log-likelihood

optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)
model.to(device)

val_acc_history = []
val_loss_history = []
train_acc_history = []
train_loss_history = []
precision_history = []
recall_history = []
f1_history = []
fp_history = []
fn_history = []

precision = 0.
recall = 0.
f1 = 0.

def validate(model):
    val_acc = 0.0
    val_loss = 0.0
    model.eval()
    running_precision = []
    running_recall = []
    running_f1 = []
    fp = 0
    fn = 0

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

            for j in range(len(labels.cpu().numpy())):
                fp_bool = labels.cpu().numpy()[j] == 0 and preds.cpu().numpy()[j] == 1
                fn_bool = labels.cpu().numpy()[j] == 1 and preds.cpu().numpy()[j] == 0
                fp += fp_bool
                fn += fn_bool
                img_paths = batch["img_path"]
                if fp_bool:
                    txt_path = modelname[:16] + 'fp.txt'
                    with open(txt_path, "a") as file:
                        file.write(f"{img_paths[j]}\n")
                if fn_bool:
                    txt_path = modelname[:16] + 'fn.txt'
                    with open(txt_path, "a") as file:
                        file.write(f"{img_paths[j]}\n")

            precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy())
            recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy())
            f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy())
            running_precision.append(precision)
            running_recall.append(recall)
            running_f1.append(f1)
        
        print(f"false positives: {fp}")
        print(f"false negatives: {fn}")

        fp_history.append(fp)
        fn_history.append(fn)

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
        
        len_train_loader = int(np.ceil(len(dset) * (1 - valid_size)))
        len_val_loader = int(np.floor(len(dset) * valid_size))
        print(len(train_loader.sampler))
        print(len_train_loader)
        print(len(val_loader.sampler))
        print(len_val_loader)

        train_acc /= len_train_loader
        train_loss /= len_train_loader

        val_acc /= len_val_loader
        val_loss /= len_val_loader

        scheduler.step(val_loss)

        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)
        
        wandb.log({"train loss": train_loss, "val loss": val_loss, "train acc": train_acc, "val acc": val_acc, "precision": precision_history[-1], "recall": recall_history[-1], "f1 score": f1_history[-1]}, step=epoch)

        time_elapsed = time.time() - since
        print("Duration: {:.0f}h {:.0f}min {:.0f}s".format(time_elapsed // 3600, (time_elapsed % 3600 // 60), time_elapsed % 60),
              "Epoch {}/{}".format(epoch+1, num_epochs),
              "Train Accuracy: {:.3f}".format(train_acc),
              "Validation Accuracy: {:.3f}".format(val_acc),
              "Train Loss: {:.3f}".format(train_loss),
              "Validation Loss: {:.3f}".format(val_loss),
              "Precision: {:.3f}".format(precision_history[-1]),
              "Recall: {:.3f}".format(recall_history[-1]),
              "F1 Score: {:.3f}".format(f1_history[-1])
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

    plt.figure(6)
    plt.plot(fp_history, label="False positives")
    plt.plot(fn_history, label="False negatives")
    plt.grid()
    plt.legend(frameon=False)

    plt.show()


if __name__ == "__main__":
    since = time.time()

    train(num_epochs)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}min {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600 // 60), time_elapsed % 60))

    torch.save(model, modelname)
    plot_metrics()
