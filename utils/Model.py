from utils.Network import ResNet18
import torch

def getModel():
    return ResNet18()

def loadModel(PATH):
    model = ResNet18()
    model.load_state_dict(torch.load(PATH))
    model.eval()

    return model