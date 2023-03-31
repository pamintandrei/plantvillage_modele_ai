from torch import utils
from torchvision.datasets.folder import ImageFolder
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

WEIGHTS = [
    0.100300903,
    0.067658999,
    0.1,
    0.1,
    0.657894737,
    0.047014575,
    0.1,
    0.052383447,
    0.105042017,
    0.056465274,
    0.039665871,
    0.071225071,
    0.009162356,
    0.268096515,
    0.012853551,
]

def load_dataset(img_size, folder, batch_size, is_val = False):
    transforms = get_preprocessing_transforms(img_size) if not is_val else get_val_preprocessing(img_size)
    dataset = ImageFolder(folder, transform=transforms)
    sampler = utils.data.WeightedRandomSampler([WEIGHTS[img[1]] for img in dataset], len(dataset),replacement=True) if not is_val else None
    data_loader = utils.data.DataLoader(
        dataset,
        sampler = sampler,
        batch_size=int(batch_size),
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    return data_loader

def get_preprocessing_transforms(size):
    t=[]
    t.append(transforms.CenterCrop(size))
    t.append(transforms.ToTensor())
    t.append(transforms.RandomHorizontalFlip(0.5))
    t.append(transforms.RandomVerticalFlip(0.5))
    t.append(transforms.ColorJitter(0.1,0.1,0.1))
    t.append(transforms.Normalize([0.485, 0.456, 0.406],  [0.229, 0.224, 0.225] ))
    t=transforms.Compose(t)
    return t

def get_val_preprocessing(size):
    t=[]
    t.append(transforms.CenterCrop(size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize([0.485, 0.456, 0.406],  [0.229, 0.224, 0.225] ))
    t=transforms.Compose(t)
    return t

#load_dataset(224, "D:/plantvillage_modele_ai/PlantVillage/", 32)

    