import torch
import torch.nn as nn

def pgd_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()

    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image

def PGD(inputs, labels, model, epsilon=8/255, stepSize=2/255, lossFun=nn.CrossEntropyLoss(), iterationNumber=10):
    adv_inputs = inputs.clone().detach().requires_grad_(True)
    outputs = model(adv_inputs)

    loss = lossFun(outputs, labels)
    loss.backward()

    data_grad = adv_inputs.grad.data
    model.zero_grad()

    for _ in range(iterationNumber):
        adv_inputs = pgd_attack(adv_inputs, epsilon, data_grad)

    return adv_inputs