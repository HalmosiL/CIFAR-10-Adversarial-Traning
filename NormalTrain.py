import torch
import torch.optim as optim
import torch.nn as nn

from Dataset import getTrainset, getTestset
from Attack import fgsm_attack
from Model import getModel

#CONFIGS 
DEVICE = "cuda:0"

BATCH_SIZE_TEST = 256
BATCH_SIZE_TRAIN = 256

EPOCHS = 50

NOM_WORKERS_TRAIN = 8
NOM_WORKERS_TEST = 8

model = getModel().to(DEVICE)

trainloader = getTrainset(BATCH_SIZE_TEST, NOM_WORKERS_TRAIN, PIN_MEMORY=True)
testloader = getTestset(BATCH_SIZE_TEST, NOM_WORKERS_TEST, PIN_MEMORY=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

TRAIN_LEN = EPOCHS * len(trainloader)

for epoch in range(EPOCHS):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0 and i != 0:
            fin = str((epoch * len(trainloader) + i) / TRAIN_LEN * 100)[:5]
            print(f'EPOCH: [{epoch + 1}] loss: {running_loss / 100:.3f} Finished: {fin}%')
            running_loss = 0.0

print('Finished Training')
print('Save model:')

torch.save(model.state_dict(), "./Models/model.pt")

correct = 0
total = 0

model = model.eval()

with torch.no_grad():
    for data in testloader:
        images, labels = data

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
