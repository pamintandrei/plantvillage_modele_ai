from torch import utils
from torch import tensor
from torchvision.datasets.folder import ImageFolder, default_loader
import torchvision.transforms as transforms
import os
import os.path
from typing import Dict, Tuple, List

def load_dataset(img_size, folder, batch_size):
    transforms = get_preprocessing_transforms(img_size)
    dataset = ImageFolder(folder,transform=transforms)
    ImageFolder.find_classes=overwrite_find_classes_attack
    data_loader = utils.data.DataLoader(
        dataset,
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

def overwrite_find_classes_attack(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    print(class_to_idx)
    return classes, class_to_idx

load_dataset(224, "D:/plantvillage_modele_ai/PlantVillage/", 32)

    