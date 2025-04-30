from torchvision import transforms
import torchvision.transforms.functional as F
import random
import torch

def resize(img, target, size=(512, 512)):
    img = F.resize(img, size)
    masks = F.resize(target["masks"].unsqueeze(1).float(), size, interpolation=F.InterpolationMode.NEAREST).squeeze(1).byte()
    target["masks"] = masks

    boxes = []
    new_masks = []
    for m in masks:
        pos = torch.where(m)
        if len(pos[0]) == 0 or len(pos[1]) == 0:
            continue 

        xmin = torch.min(pos[1])
        xmax = torch.max(pos[1])
        ymin = torch.min(pos[0])
        ymax = torch.max(pos[0])

        if xmax - xmin < 1 or ymax - ymin < 1:
            continue  

        boxes.append(torch.tensor([xmin, ymin, xmax, ymax]))
        new_masks.append(m)

    if len(boxes) == 0:
        boxes = [torch.tensor([0, 0, 1, 1])]
        new_masks = [torch.zeros_like(masks[0])]

    target["boxes"] = torch.stack(boxes).float()
    target["masks"] = torch.stack(new_masks)
    return img, target


def transform(img, target, train=True):
    img, target = resize(img, target, size=(512, 512))

    if train:
        if random.random() > 0.5:
            img = F.hflip(img)
            target["masks"] = target["masks"].flip(-1)
            target["boxes"][:, [0, 2]] = img.shape[2] - target["boxes"][:, [2, 0]]

        if random.random() > 0.5:
            img = F.vflip(img)
            target["masks"] = target["masks"].flip(-2)
            target["boxes"][:, [1, 3]] = img.shape[1] - target["boxes"][:, [3, 1]]

        if random.random() < 0.3:
            img = F.gaussian_blur(img, kernel_size=3)

        color_aug = transforms.ColorJitter(brightness=0.2, contrast=0.2)
        img = color_aug(img)

    return img, target


def get_transform(train):
    return lambda img, target: transform(img, target, train)



transform_val = transforms.Compose([
            transforms.ToTensor(),
])
