import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import visdom
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

tb = SummaryWriter()


torch.set_printoptions(linewidth=120)

data_dir = 'drive_day_2019_08_21_16_14_06/test2'

def load_split_train_test(datadir, valid_size=0.1):
    train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       ])
    test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      ])
    train_data = datasets.ImageFolder(datadir,       
                    transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,
                transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=1) # 64
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=1) # 64

    return trainloader, testloader

trainloader, testloader = load_split_train_test(data_dir, 0.2)
#print(trainloader.dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
model = models.resnet18(pretrained=True)  # on ImageNet
#print(model)

images, labels = next(iter(trainloader))
grid = torchvision.utils.make_grid(images)
tb.add_image("images", grid)
tb.add_graph(model, images)
tb.close()

for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(512, 512), # 2048, 512
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()  # negative log-likelihood
optimizer = optim.SGD(model.fc.parameters(), lr=0.003)
# optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)

# training accuracy anschauen

# tensorboardx
# visdom
# weightandbiases

class Visualizations:
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.loss_win = None
        self.loss_win2 = None

    def plot_loss(self, loss, step):
        self.loss_win = self.vis.line(
            [loss],
            [step],
            win=self.loss_win,
            update='append' if self.loss_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Loss',
                title='Train Loss',
            )
        )
    def plot_loss2(self, loss, step):
        self.loss_win2 = self.vis.line(
            [loss],
            [step],
            win=self.loss_win2,
            update='append' if self.loss_win2 else None,
            opts=dict(
                xlabel='Step',
                ylabel='Loss',
                title='Test Loss',
            )
        )

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

epochs = 1
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
#vis = visdom.Visdom()
vis = Visualizations()

'''
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            test_accuracy = 0
            train_accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    #print(logps)
                    #print('-----')
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            vis.plot_loss(np.mean(running_loss), steps)
            vis.plot_loss2(np.mean(test_loss), steps)

            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {test_accuracy/len(testloader):.8f}")
            running_loss = 0
            model.train()
'''
torch.save(model, 'activemodel17.pth')


plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.grid()
plt.legend(frameon=False)
plt.show()

