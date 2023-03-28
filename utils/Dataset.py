import torchvision.transforms as transforms
import torchvision
import torch

transform_train =  transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test =  transforms.Compose([
    transforms.ToTensor(),
])

def getTrainset(BATCH_SIZE_TRAIN, NOM_WORKERS_TRAIN, PIN_MEMORY=False):
    trainset = torchvision.datasets.CIFAR10(
        root='../Data',
        train=True,
        download=True,
        transform=transform_train
    )

    return torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=NOM_WORKERS_TRAIN,
        pin_memory=PIN_MEMORY
    )

def getTestset(BATCH_SIZE_TEST, NOM_WORKERS_TEST, PIN_MEMORY=False):
    testset = torchvision.datasets.CIFAR10(
        root='../Data',
        train=False,
        download=True,
        transform=transform_test
    )

    return torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=NOM_WORKERS_TEST,
        pin_memory=PIN_MEMORY
    )