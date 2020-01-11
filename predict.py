import os
import sys
import shutil
import cv2
from PIL import Image, ImageTk
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision
import time
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from IRDataset import IRDataset, IRDatasetTest
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, roc_curve, precision_score, recall_score, f1_score

since = time.time()

dir_n = 'drive_all_night_test_20_backup'
dir_d = 'drive_all_day_test_20_backup'
dir_c = 'drive_all_test_20_backup'

# 1/5/9/12 night
# 2/4/6/7/8/10 day
# 3/11 combined
####
MODELPATH = "/home/viki/Documents/Informatik/BA/ActiveCarModel6.pth"
data_dir = dir_c
####

mean_0 = [0.6968, 0.6968, 0.6968]
std_0 = [0.2580, 0.2580, 0.2580]

mean_1 = [0.5094, 0.5094, 0.5094]
std_1 = [0.3474, 0.3474, 0.3474]

mean_2 = [0.5432, 0.5432, 0.5432]
std_2 = [0.3317, 0.3317, 0.3317]

if data_dir == dir_n:
    mean, std = mean_0, std_0
if data_dir == dir_d:
    mean, std = mean_1, std_1
if data_dir == dir_c:
    mean, std = mean_2, std_2

def load_model(MODELPATH):
    model = torchvision.models.resnet18(num_classes=2)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(#nn.Linear(num_features, 512),
                             #nn.LeakyReLU(0.3), 
                             #nn.Dropout(0.1), 
                             nn.Linear(num_features, 2), 
                             nn.LogSoftmax(dim=1))

    state_dict = torch.load(MODELPATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model

def fill_beneath_step(x, y, color, alpha=0.2):
    x_long = [v for v in x for _ in (0, 1)][:-1]
    y_long = [v for v in y for _ in (0, 1)][1:]
    plt.fill_between(x_long, 0, y_long, alpha=alpha, color=color)


image_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std),
                                   ])

dset = IRDatasetTest(data_dir, transform=image_transforms)
batch_size = 912

print('Number of images: ', len(dset))
print('Number of categories: ', len(dset.categories))

test_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size)
images, labels, _ = next(iter(test_loader))
model = load_model(MODELPATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calc_average_precision_score(recall, precision):
    n = len(recall)
    ap = 0
    for i in range(0, n-1):
        ap += (recall[i+1] - recall[i]) * precision[i+1]
    return abs(ap)

def calc_auc(fpr, tpr):
    n = len(fpr)
    auc = 0
    for i in range(0, n-1):
        auc += (fpr[i+1] - fpr[i]) * tpr[i+1]
    return abs(auc)

def calc_pr(outputs, labels, step):
    recalls, precisions = [], []
    steps = np.arange(0, 1, step)
    for i in steps:
        thresholds = outputs[:, 1].gt(i).int().cpu().numpy()
        precision = precision_score(labels, thresholds)
        recall = recall_score(labels, thresholds)
        recalls.append(recall)
        precisions.append(precision)
    return recalls, precisions

def calc_roc(outputs, labels, step):
    fprs, tprs = [], []
    steps = np.arange(0, 1, step)
    for i in steps:
        thresholds = outputs[:, 1].gt(i).int().cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(labels, thresholds).ravel()
        tpr = tp / (fn + tp)
        fpr = fp / (tn + fp)
        fprs.append(fpr) 
        tprs.append(tpr)
    return fprs, tprs


with torch.no_grad():
    for i, batch in enumerate(test_loader):
        images, labels = batch["image"].to(device), batch["category"].to(device)
        outputs = model(images)
        outputs = torch.exp(outputs)
        _, preds = torch.max(outputs, 1)
        probs = torch.topk(outputs, k=1)

        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()

        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        accuracy = accuracy_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds)

        print("Confusion matrix: ", tn, fp, fn, tp)
        print("Accuracy: {0:0.2f}".format(accuracy))
        print("Precision: {0:0.2f}".format(precision))
        print("Recall: {0:0.2f}".format(recall))
        print("F1: {0:0.2f}".format(f1))
        print('--------')

        modelpath_name = MODELPATH.rsplit("/")[-1]
        print(f"Classification report on {modelpath_name} with test set '{data_dir}'")
        print(classification_report(labels, preds))
        print('--------')

        recall, precision = calc_pr(outputs, labels, 0.01)
        avg_prec = sum(precision) / len(precision)
        
        recall = [el for el in recall if el != 0.0]
        precision = [el for el in precision if el != 0.0]
        recall = [1.0] + recall + [0.0]
        precision = [0.0] + precision + [1.0]

        plt.figure("PR-Curve")
        plt.step(recall, precision, label='Precision-Recall curve')
        no_skill = len(labels[labels==1]) / len(labels)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No skill')
        fill_beneath_step(recall, precision, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: AUC={0:0.2f}'.format(
                avg_prec))
        plt.legend(loc="lower left")
        print('Average precision-recall score: {0:0.2f}'.format(avg_prec))

        ns_probs = [0 for _ in range(len(probs[0]))]
        ns_auc = roc_auc_score(labels, ns_probs)
        print('No skill: ROC AUC=%.3f' % (ns_auc))
        ns_fpr, ns_tpr, _ = roc_curve(labels, ns_probs)
        
        fpr, tpr = calc_roc(outputs, labels, 0.01)
        auc = calc_auc(fpr, tpr)
        fpr = fpr + [0.0]
        tpr = tpr + [0.0]
        print('Trained Classifier: ROC AUC=%.3f' % (auc))

        pyplot.figure("ROC-AUC")
        pyplot.title('ROC')
        plt.step(fpr, tpr, marker='.', label='Classifier (AUC = {0:0.2f})'.format(auc))
        pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No skill')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.legend(loc="lower right")

        time_elapsed = time.time() - since
        print("Duration: {:.0f}h {:.0f}min {:.0f}s".format(time_elapsed // 3600, (time_elapsed % 3600 // 60), time_elapsed % 60))

        pyplot.show()
