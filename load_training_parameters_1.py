import torch

# Training parameeters(optimizer, loss etc) for the first experiment

def load_training(parameters):
    optimizer = torch.optim.Adam(parameters, lr=0.0001)
    loss = torch.nn.CrossEntropyLoss()
    epochs = 120
    return optimizer, loss, epochs