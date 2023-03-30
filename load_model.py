from torchvision import  models
import torch.nn as nn

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

def load_model():
    model_ft = models.resnet50(pretrained=True)
    set_parameter_requires_grad(model_ft)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 15)
    return model_ft