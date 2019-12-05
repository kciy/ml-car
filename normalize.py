import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import datasets, transforms

dataset = datasets.ImageFolder('drive_all_day_train',
                               transform=transforms.Compose([
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor()]))

loader = DataLoader(dataset,
                    batch_size=10,
                    num_workers=3,
                    shuffle=False)

mean = 0.
std = 0.
for images, _ in loader:
    # batch size (the last batch can have smaller size!)
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)

mean /= len(loader.dataset)
std /= len(loader.dataset)

print('mean: ', mean)
print('std: ', std)
