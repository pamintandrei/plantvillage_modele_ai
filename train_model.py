from load_dataset import load_dataset
from load_model import load_model
from load_training_parameters_1 import load_training
from utils import write_results
import torch



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = load_model() # load_model
model.to(device)


train_loader = load_dataset(224, "E:/modele_plantvilige/PlantVillageSplit/train", 64)
test_loader = load_dataset(224, "E:/modele_plantvilige/PlantVillageSplit/val", 16, True)

optimizer, criterion, epochs = load_training(model.parameters()) # put params here

for i in range(epochs):
    loss_avg = 0
    model.train()
    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.to(device), labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_avg+= loss.cpu()
        if idx%50==0:
            print(f"done:{(idx+1)/len(train_loader)*100} loss:{loss_avg/(idx+1)}")
    print(f"done:{(idx+1)/len(train_loader)*100} loss:{loss_avg/(idx+1)}")
    write_results("results/train.csv", i+1, loss_avg/len(train_loader), 0, 0)
    
    
    loss_avg, accuracy, top5 = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            outputs = outputs.cpu()
            labels = labels.cpu()
            loss_avg += loss.cpu()
            
            top1_list = torch.topk(outputs,1)
            top5_list = torch.topk(outputs, 5)
            accuracy += sum([labels[idx] in top1_list[1][idx] for idx in range(len(images))]) / len(images)
            top5 += sum([labels[idx] in top5_list[1][idx] for idx in range(len(images))]) / len(images)

        
    print(
        f"Validation after epoch {i+1}: loss={loss_avg/len(test_loader)} "
        f"accuracy={accuracy/len(test_loader)} top5={top5/len(test_loader)}"
    )
    write_results(
        "results/validation.csv", epoch=i+1, loss=loss_avg/len(test_loader), accuracy=accuracy/len(test_loader),
        top5 = top5/len(test_loader)
    )
    if i%5==0:
        torch.save(model.state_dict(), f"checkpoint/resnet101_{i}")

