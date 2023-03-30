import torch

# Training parameeters(optimizer, loss etc) for the first experiment

def load_training(parameters):
    optimizer = torch.optim.AdamW(parameters, lr=0.001)
    loss = torch.nn.CrossEntropyLoss()
    epochs = 20
    return optimizer, loss, epochs