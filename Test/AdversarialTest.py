from autoattack import AutoAttack
from Model import loadModel
from Dataset import getTestset

import torch

DEVICE = "cuda:0"
BATCH_SIZE_TEST = 256
NOM_WORKERS_TEST = 12

TEST_BATCH_NUM = 1

model = loadModel(PATH="./Models/model.pt").to(DEVICE).eval()

adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard')
testloader = getTestset(BATCH_SIZE_TEST, NOM_WORKERS_TEST, PIN_MEMORY=True)

correct = 0
total = 0

for i, data in enumerate(testloader):
    images, labels = data

    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    x_adv = adversary.run_standard_evaluation(images, labels, bs=BATCH_SIZE_TEST)

    outputs = model(x_adv)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    
    if(TEST_BATCH_NUM == i + 1):
        break

print(f'Accuracy of the network on the {TEST_BATCH_NUM * BATCH_SIZE_TEST} test images: {100 * correct // total} %')