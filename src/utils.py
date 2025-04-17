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
            continue  # 跳過這個 mask

        xmin = torch.min(pos[1])
        xmax = torch.max(pos[1])
        ymin = torch.min(pos[0])
        ymax = torch.max(pos[0])

        if xmax - xmin < 1 or ymax - ymin < 1:
            continue  # 忽略寬高不合法的 box

        boxes.append(torch.tensor([xmin, ymin, xmax, ymax]))
        new_masks.append(m)

    if len(boxes) == 0:
        # 為了避免空資料導致後續崩潰，隨便填一個 dummy box 和 mask
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
    return img, target

def get_transform(train):
    # 返回全局函數，並傳遞 train 參數
    return lambda img, target: transform(img, target, train)



transform_val = transforms.Compose([
            transforms.ToTensor(),
])
