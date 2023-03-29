import torch
import os

def Test(model, testloader, criterion, Attack, SaveFolder):
    model = model.eval()

    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images = images.to(CONFIG["DEVICE"])
            labels = labels.to(CONFIG["DEVICE"])

            inputs = FGSM(
                inputs,
                labels,
                model,
                8/255,
                criterion
            )

MODEL_PATH = "../normal_MaxEpoch_150_BatchSize_128/"
SAVE_FOLDER = "../Transfer/" + MODEL_PATH[3:] 
