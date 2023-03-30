from torch import utils
from torch import tensor
from torchvision.datasets.folder import ImageFolder, default_loader
import torchvision.transforms as transforms
import os
import os.path
from typing import Dict, Tuple, List


WEIGHTS = [
    0.001003009,
    0.00067659,
    0.001,
    0.006578947,
    0.001,
    0.000470146,
    0.001,
    0.000628536,
    0.000523834,
    0.00105042,
    0.000564653,
    0.000596659,
    0.000712251,
    0.002680965,
    0.000311624,
]

def load_dataset(img_size, folder, batch_size, num_samples):
    transforms = get_preprocessing_transforms(img_size)
    dataset = ImageFolder(folder, transform=transforms)
    sampler = utils.data.WeightedRandomSampler([WEIGHTS[img[1]] for img in dataset], num_samples*batch_size)
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
    # from https://www.frontiersin.org/articles/10.3389/frai.2022.868926/full
    t.append(transforms.Normalize([0.4685, 0.5424, 0.4491],  [0.2337, 0.2420,0.2531] ))
    t.append(transforms.RandomRotation(90))
    t.append(transforms.RandomHorizontalFlip(0.5))
    t.append(transforms.RandomVerticalFlip(0.5))
    t=transforms.Compose(t)
    return t

#load_dataset(224, "D:/plantvillage_modele_ai/PlantVillage/", 32)

    