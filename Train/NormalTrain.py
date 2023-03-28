import torch
import json
import sys
import tqdm
import os
import torch.optim as optim
import torch.nn as nn

from utils.Dataset import getTrainset, getTestset
from utils.Attack import fgsm_attack
from utils.Model import getModel

CONFIG = json.load(open(sys.argv[1]))

NOM_WORKERS_TRAIN = CONFIG["NOM_WORKERS_TRAIN"]
NOM_WORKERS_TEST = CONFIG["NOM_WORKERS_TEST"]

NAME = CONFIG["NAME"] + "_MaxEpoch_" + str(CONFIG["EPOCHS"]) + "_BatchSize_" + str(CONFIG["BATCH_SIZE_TRAIN"])
SAVE_PATH = "./Models/" + NAME

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

model = getModel().to(CONFIG["DEVICE"])

trainloader = getTrainset(CONFIG["BATCH_SIZE_TRAIN"], NOM_WORKERS_TRAIN, PIN_MEMORY=True)
testloader = getTestset(CONFIG["BATCH_SIZE_TEST"], NOM_WORKERS_TEST, PIN_MEMORY=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in tqdm.tqdm(range(CONFIG["EPOCHS"])):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        inputs = inputs.to(CONFIG["DEVICE"])
        labels = labels.to(CONFIG["DEVICE"])

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    if(epoch % CONFIG["ModelSavePeriod"] == 0):
        torch.save(model.state_dict(), SAVE_PATH + f"/model_{epoch}.pt")

print('Finished Training')
print('Save model:')

torch.save(model.state_dict(), SAVE_PATH + "/model_fin.pt")

correct = 0
total = 0

model = model.eval()

with torch.no_grad():
    for data in testloader:
        images, labels = data

        images = images.to(CONFIG["DEVICE"])
        labels = labels.to(CONFIG["DEVICE"])

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
