from load_dataset import load_dataset
from load_model import load_model
from load_training_parameters_1 import load_training
from utils import write_results
import torch


model = load_model() # load_model

train_loader = load_dataset(224, "D:\plantvillage_modele_ai\PlantVillageSplit\train", 64)
test_loader = load_dataset(224, "D:\plantvillage_modele_ai\PlantVillageSplit\train", 64)

optimizer, criterion, epochs = load_training() # put params here

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i in range(epochs):
    loss_avg = 0
    for images,labels in train_loader:
        images.to(device)
        labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_avg+= loss.cpu()    
    write_results("train.csv",0,0, epoch=i+1, loss=loss_avg/len(train_loader))
    
    
    loss_avg, accuracy, top5 = 0
    for images, labels in test_loader:
        images.to(device)
        labels.to(device)
        
        # Forward pass
        outputs = model(images).cpu()
        loss = criterion(outputs, labels)
        loss_avg += loss.cpu()
        top1_list = torch.topk(outputs,1)
        top5_list = torch.topk(outputs, 5)
        
        accuracy += sum([int(labels[idx] in top1_list[idx]) for idx in len(images)]) / len(images)
        top5 += sum([int(labels[idx] in top5_list[idx]) for idx in len(images)]) / len(images)
    print(
        f"Validation after epoch {i+1}: loss={loss_avg/len(test_loader)}"
        f"accuracy={accuracy/len(test_loader)} top5={top5/len(test_loader)}"
    )
    write_results(
        "validation.csv", epoch=i+1, loss=loss_avg/len(test_loader), acc=accuracy/len(test_loader),
        top5 = top5/len(test_loader)
    )
        
