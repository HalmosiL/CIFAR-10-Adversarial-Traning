import torch
import json
import sys
import tqdm
import os
import sys

import wandb

import torch.optim as optim
import torch.nn as nn

sys.path.insert(1, '../')

from utils.Dataset import getTrainset, getTestset
from utils.Attack import PGD
from utils.Model import getModel

def Train(model, optimizer, trainloader, criterion, scheduler):
    model = model.train()

    correct = 0
    total = 0
    loss_log = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        inputs = inputs.to(CONFIG["DEVICE"])
        labels = labels.to(CONFIG["DEVICE"])
        
        inputs = PGD(inputs, labels, model)

        optimizer.zero_grad()

        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_log += loss.item() / trainloader.__len__()

    if scheduler is not None:
        scheduler.step()

    wandb.log({"train_acc_adversarial": 100 * correct / total, "train_loss_adversarial": loss_log})

    return model

def TestAdversarial(model, testloader, criterion):
    correct = 0
    total = 0
    loss = 0

    model = model.eval()

    for data in testloader:
        images, labels = data

        images = images.to(CONFIG["DEVICE"])
        labels = labels.to(CONFIG["DEVICE"])

        images = PGD(images, labels, model)

        outputs = model(images)

        loss += criterion(outputs, labels).item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    loss = loss / testloader.__len__()
    wandb.log({"val_acc_adversarial": 100 * correct / total, "val_loss_adversarial": loss})

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

def Test(model, testloader, criterion):
    correct = 0
    total = 0
    loss = 0

    model = model.eval()

    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images = images.to(CONFIG["DEVICE"])
            labels = labels.to(CONFIG["DEVICE"])

            outputs = model(images)

            loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        loss = loss / testloader.__len__()

    wandb.log({"val_acc": 100 * correct / total, "val_loss": loss})
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')


def schale(epoch):
    if(epoch <= 100):
        return 1
    
    if(epoch <= 150):
        return 0.1

    return 0.01

CONFIG = json.load(open(sys.argv[1]))

NOM_WORKERS_TRAIN = CONFIG["NOM_WORKERS_TRAIN"]
NOM_WORKERS_TEST = CONFIG["NOM_WORKERS_TEST"]

NAME = "Adversarial" + CONFIG["NAME"] + "_MaxEpoch_" + str(CONFIG["EPOCHS"]) + "_BatchSize_" + str(CONFIG["BATCH_SIZE_TRAIN"])
SAVE_PATH = "../Models/" + NAME

wandb.init(
    project="CIFAR-10-Adversarial-Traning",

    config={
    "learning_rate": 0.1,
    "architecture": "RESNET18",
    "dataset": "CIFAR-10",
    "epochs": CONFIG["EPOCHS"],
    },

    name=NAME
)

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

model = getModel().to(CONFIG["DEVICE"])

trainloader = getTrainset(CONFIG["BATCH_SIZE_TRAIN"], NOM_WORKERS_TRAIN, PIN_MEMORY=True)
testloader = getTestset(CONFIG["BATCH_SIZE_TEST"], NOM_WORKERS_TEST, PIN_MEMORY=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[schale])

for epoch in tqdm.tqdm(range(CONFIG["EPOCHS"])):
    model = Train(model, optimizer, trainloader, criterion, scheduler)
    
    if(epoch % CONFIG["ModelSavePeriod"] == 0):
        TestAdversarial(model, testloader, criterion)
        Test(model, testloader, criterion)
    
    if(epoch % CONFIG["ModelSavePeriod"] == 0):
        torch.save(model.state_dict(), SAVE_PATH + f"/model_{epoch}.pt")

print('Finished Training')
print('Save model:')

torch.save(model.state_dict(), SAVE_PATH + "/model_fin.pt")

