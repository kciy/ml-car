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
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, plot_precision_recall_curve, average_precision_score
from IRDataset import IRDataset, IRDatasetTest
import time
import datetime
import wandb


dir_0 = 'drive_all_night'
dir_1 = 'drive_all_day'
dir_2 = 'drive_all'

#####
data_dir = dir_2
modelname = 'ActiveCarModel14.pth'
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

image_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std),
                                   ])

print(f'Starting time: {datetime.datetime.now()}')

num_epochs = 100
batch_size = 64
test_size = 0.2

wandb.config.epochs = num_epochs
wandb.config.batch_size = batch_size

# load data
def load_split_train_test(datadir, size=0.2):
    train_data = IRDataset(data_dir, transform=image_transforms)
    test_data = IRDatasetTest(data_dir, transform=image_transforms)
    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(size * num_train))
    np.random.shuffle(indices)

    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_loader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=batch_size)

    return train_loader, test_loader, train_data, test_data

dset = IRDataset(data_dir, transform=image_transforms)
print('Number of images: ', len(dset))
print('Number of categories: ', len(dset.categories))

train_loader, test_loader, train_data, test_data = load_split_train_test(data_dir, test_size)
images, labels, _ = next(iter(train_loader))
model = torchvision.models.resnet18(pretrained=True)
'''
X_train, X_test, y_train, y_test = [], [], [], []
for i in range(len(train_data)):
    X_train.append(train_data[i]["image"])
    y_train.append(train_data[i]["category"])
for i in range(len(test_data)):
    X_test.append(test_data[i]["image"])
    y_test.append(test_data[i]["category"])
'''
#print(X_train, y_train)
#print(X_test, y_test)

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
#for param in model.parameters():
#    param.requires_grad = False

num_features = model.fc.in_features                    
model.fc = nn.Sequential(nn.Linear(num_features, 2), nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss().to(device)  # negative log-likelihood

optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
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
fpr_history = []
fn_history = []
tp_history = []
tn_history = []

def validate(model):
    val_acc = 0.0
    val_loss = 0.0
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    running_recall = []

    running_precision = []
    running_f1 = []

    model.eval()
    with torch.no_grad():
        ii = 0
        for i, batch in enumerate(test_loader):
            ii += 1
            print(ii, end='\r')
            images, labels = batch["image"].to(device), batch["category"].to(device)
            outputs = model(images)
            #outputs = torch.exp(outputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_acc += torch.sum(labels.data == preds).item()

            for j in range(len(labels.cpu().numpy())):
                tn_bool = labels.cpu().numpy()[j] == 0 and preds.cpu().numpy()[j] == 0
                fp_bool = labels.cpu().numpy()[j] == 0 and preds.cpu().numpy()[j] == 1
                fn_bool = labels.cpu().numpy()[j] == 1 and preds.cpu().numpy()[j] == 0
                tp_bool = labels.cpu().numpy()[j] == 1 and preds.cpu().numpy()[j] == 1
        
                fp += fp_bool
                fn += fn_bool
                tp += tp_bool
                tn += tn_bool

                img_paths = batch["img_path"]
                
            running_precision.append(precision_score(labels.cpu().numpy(), preds.cpu().numpy()))    
            running_recall.append(recall_score(labels.cpu().numpy(), preds.cpu().numpy()))     
            running_f1.append(f1_score(labels.cpu().numpy(), preds.cpu().numpy()))     

            precision, recall, _ = precision_recall_curve(labels.cpu().numpy(), preds.cpu().numpy())
            print(precision, recall)
            print(running_precision, running_recall)
            print('----------')
            precision_history.append(precision_score(labels.cpu().numpy(), preds.cpu().numpy()))
            recall_history.append(recall_score(labels.cpu().numpy(), preds.cpu().numpy()))
            f1_history.append(f1_score(labels.cpu().numpy(), preds.cpu().numpy()))

            fpr = fp / (fp + tn)
        
        print(f"false positives: {fp}")
        print(f"false negatives: {fn}")
        print(f"true positives: {tp}")
        print(f"true negatives: {tn}")
        print(f"false positive rate: {fpr}")

        fp_history.append(fp)
        fn_history.append(fn)
        tp_history.append(tp)
        tn_history.append(tn)
        fpr_history.append(fpr)

        return val_loss, val_acc, running_recall, running_precision, running_f1

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
            #outputs = torch.exp(outputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            step += 1
            
            train_loss += loss.item() * images.size(0)
            train_acc += torch.sum(preds == labels.data).item() 
        
        print("\n")
        val_loss, val_acc, recall, precision, f1 = validate(model)
        
        len_train_loader = int(np.ceil(len(dset) * (1 - test_size)))
        len_test_loader = int(np.floor(len(dset) * test_size))

        train_acc /= len_train_loader
        train_loss /= len_train_loader

        val_acc /= len_test_loader
        val_loss /= len_test_loader

        scheduler.step(val_loss)
        recall = np.mean(recall)
        precision = np.mean(precision)
        f1 = np.mean(f1)

        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)
        
        wandb.log({"train loss": train_loss, "val loss": val_loss, "train acc": train_acc, "val acc": val_acc, "recall": recall, "precision": precision} , step=epoch)

        time_elapsed = time.time() - since
        print("Duration: {:.0f}h {:.0f}min {:.0f}s".format(time_elapsed // 3600, (time_elapsed % 3600 // 60), time_elapsed % 60),
              "Epoch {}/{}".format(epoch+1, num_epochs),
              "Train Accuracy: {:.3f}".format(train_acc),
              "Validation Accuracy: {:.3f}".format(val_acc),
              "Train Loss: {:.3f}".format(train_loss),
              "Validation Loss: {:.3f}".format(val_loss),
              "Precision: {:.3f}".format(precision),
              "Recall: {:.3f}".format(recall),
              "F1 Score: {:.3f}".format(f1)
             )

def plot_metrics():
    plt.figure("Accuracy")
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.grid()
    plt.legend(frameon=False)

    plt.figure("Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.plot(train_loss_history, label="Train Loss")
    plt.grid()
    plt.legend(frameon=False)

    plt.figure("Precision")
    plt.plot(precision_history, label="Precision")
    plt.grid()
    plt.legend(frameon=False)
    
    plt.figure("Recall")
    plt.plot(recall_history, label="Recall")
    plt.grid()
    plt.legend(frameon=False)
    
    plt.figure("F1")
    plt.plot(f1_history, label="F1 Score")
    plt.grid()
    plt.legend(frameon=False)

    plt.figure("FPR")
    plt.plot(fpr_history, label="false positive rate")
    plt.grid()
    plt.legend(frameon=False)

    plt.show()


if __name__ == "__main__":
    since = time.time()

    train(num_epochs)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}min {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600 // 60), time_elapsed % 60))
    torch.save(model, modelname)
    
    from numpy import asarray, savetxt
    j = 0
    for hist_name in [fp_history, fn_history, tp_history, tn_history, precision_history, recall_history, f1_history, fpr_history]:
        data = asarray(hist_name)
        filename = str(j) + ".txt"
        savetxt(filename, data)
        j += 1
    plot_metrics()
