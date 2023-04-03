import torch
from matplotlib import pyplot as plt
from sklearn import metrics

from class_to_idx import cls_to_idx
from load_dataset import load_dataset
from load_model_from_checkpoint import device, model


def generate_confusion_matrix():
    train_loader = load_dataset(224, "C:/modele_plantvilige/PlantVillageSplit/train", 64)

    label_list = []
    output_list = []

    for idx, (images, labels) in enumerate(train_loader):
        print(idx, "/", len(train_loader))
        images = images.to(device)
        label_list.extend(labels.data)

        output_list.extend((torch.max(torch.exp(model(images)), 1)[1]).data.cpu())

    matrix = metrics.confusion_matrix(label_list, output_list)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=list(cls_to_idx.keys()))
    cm_display.plot()
    plt.show()


generate_confusion_matrix()
