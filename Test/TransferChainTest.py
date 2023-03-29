import torch
import os
import sys
import torch.nn as nn

sys.path.insert(1, '../')

from utils.Attack import FGSM
from utils.Dataset import getTrainset
from utils.Model import loadModel

def GenerateImages(model, loader, criterion, saveFolder, device):
    model = model.eval()

    with torch.no_grad():
        for i, data in enumerate(loader):
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            images = FGSM(
                images,
                labels,
                model,
                8/255,
                criterion
            )

MODEL_PATH = "../Models/normal_MaxEpoch_150_BatchSize_128/"
SAVE_FOLDER = "../Transfer/" + MODEL_PATH.split("/")[2] + "/"

BATCH_SIZE = 256
NOM_WORKERS = 24
TEST_PERIOD = 10
DEVICE = "cuda:0"

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

loader = getTrainset(BATCH_SIZE, NOM_WORKERS, PIN_MEMORY=True)
criterion = nn.CrossEntropyLoss()

for i in range(TEST_PERIOD):
    model = loadModel(MODEL_PATH + f"model_{i}.pt").to(DEVICE)

    saveFolder = SAVE_FOLDER + f"model_{i}"

    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    GenerateImages(
        model,
        loader,
        criterion,
        saveFolder,
        DEVICE
    )