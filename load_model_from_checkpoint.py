import torch
from load_model import load_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model() # load_model
model.to(device)
model.load_state_dict(torch.load('checkpoint/resnet50_5'))