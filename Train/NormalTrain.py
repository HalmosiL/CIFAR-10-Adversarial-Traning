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
from utils.Model import getModel

os.environ["WANDB_RUN_GROUP"] = "Normal"

def Train(model, optimizer, trainloader, criterion, scheduler, step):
    model = model.train()

    correct = 0
    total = 0
    loss_log = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        inputs = inputs.to(CONFIG["DEVICE"])
        labels = labels.to(CONFIG["DEVICE"])

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

    wandb.log({"train_acc": 100 * correct / total, "train_loss": loss_log}, step=step)

    return model

def Test(model, testloader, criterion, step):
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

    wandb.log({"val_acc": 100 * correct / total, "val_loss": loss}, step=step)

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

NAME = CONFIG["NAME"] + "_MaxEpoch_" + str(CONFIG["EPOCHS"]) + "_BatchSize_" + str(CONFIG["BATCH_SIZE_TRAIN"])
SAVE_PATH = "../Models/" + NAME

wandb.init(
    project="CIFAR-10-Adversarial-Traning",
    group="Normal",
    job_type="Train",
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
    model = Train(model, optimizer, trainloader, criterion, scheduler, epoch + 1)
    
    if(epoch % CONFIG["ModelSavePeriod"] == 0):
        Test(model, testloader, criterion, epoch + 1)
    
    if(epoch % CONFIG["ModelSavePeriod"] == 0):
        torch.save(model.state_dict(), SAVE_PATH + f"/model_{epoch}.pt")

print('Finished Training')
print('Save model:')

torch.save(model.state_dict(), SAVE_PATH + "/model_fin.pt")

