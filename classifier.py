import torch
import torch.utils.data as data
import numpy as np
import cv2
import random
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

class Dataset(data.Dataset):
    def __init__(self, root, test = False):
        super(Dataset, self).__init__()
        self.root = root
        self.input1 = []
        self.input2 = []
        self.labels = []
        if not test:
            path = "labels.txt"
        else:
            path = "valid_labels.txt"
        with open(path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip().split(" ")
                self.input1.append(line[0])
                self.input2.append(line[1])
                self.labels.append(float (line[2]))
        self.input1 = np.array(self.input1)
        self.input2 = np.array(self.input2)
        self.labels = np.array(self.labels)

    def __getitem__(self, index):
        inp1 = torch.Tensor(np.load(f"{self.root}/{self.input1[index]}"))
        inp2 = torch.Tensor(np.load(f"{self.root}/{self.input2[index]}"))
        label = torch.Tensor([self.labels[index]])
        return inp1, inp2, label
    
    def __len__(self):
        return len(self.labels)

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(42804, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(42804, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x1, x2):
        x1 = self.layer1(x1)
        x2 = self.layer1(x2)
        
        x = torch.cat((x1, x2), 1)

        x = self.classifier(x)

        return x

def cal_acc(label, pred):
    pred = (pred > 0.5)
    label = (label == 1)
    cmp = (label == pred)
    acc = torch.count_nonzero(cmp)/len(label)
    return acc

training_data = Dataset("features/")
trainloader = torch.utils.data.DataLoader(training_data, batch_size=128, shuffle=True, num_workers=4)
validset = Dataset("validset/", test=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=4, shuffle=True, num_workers=4)

model = CustomModel()
model.to("cuda")

optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999), weight_decay = 5e-4)
criterion = nn.BCELoss()

EPOCHS = 100
best_vald_acc = -1

for epoch in range(EPOCHS):
    pbar = tqdm(trainloader, total = len(trainloader))
    for x1, x2, label in pbar:
        model.train()
        optimizer.zero_grad(True)
        label = label.to("cuda")
        output = model(x1.to("cuda"), x2.to("cuda"))
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"training epoch: {epoch} loss: {round(loss.item(), 4)}")
    
    model.eval()
    with torch.no_grad():
        pbar = tqdm(validloader, total = len(validloader))
        ACC = 0
        for x1, x2, label in pbar:
            label = label.to("cuda")
            output = model(x1.to("cuda"), x2.to("cuda"))

            acc = cal_acc(label, output)
            pbar.set_description(f"valid Epoch: {epoch} Acc: {acc}")
            ACC += acc.item()
        ACC = ACC / len(validloader)
        if ACC > best_vald_acc:
            best_vald_acc = ACC
            torch.save(model.state_dict(), f"save_dir/epoch_{epoch}_acc_{round(ACC, 2)}.pth")
        print(f"===========> ACC: {ACC}")