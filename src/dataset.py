import os
import json
from json import load

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import torch
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

def show_sample(dataset, index=None):
    if index is None:
        index = random.randint(0, len(dataset)-1)

    img, target = dataset[index]
    img_np = img.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(img_np)
    for i, mask in enumerate(target["masks"]):
        if i < 125:
            continue
        masked = np.ma.masked_where(mask == 0, mask)
        # plt.imshow(masked, alpha=0.4, cmap="jet")
        box = target["boxes"][i]
        plt.gca().add_patch(
            plt.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor='lime',
                facecolor='none'
            )
        )
        plt.text(box[0], box[1]-5, f"ID {i}", color="lime", fontsize=12)
    plt.title(f"Sample {index} with {len(target['masks'])} instances")
    plt.axis("off")
    plt.show()


class TrainDatasets(Dataset):
    def __init__(self, imgdir, json_path, transform=None):
        self.imgdir = imgdir
        self.transform = transform
        # json_file = open(json_path,'r')
        # f =  json_file.read()   # 要先使用 read 讀取檔案
        # name_to_id_tmp = json.loads(f) 
        # name_to_id = {}
        # for tmp in name_to_id_tmp:
        #     name_to_id[tmp["file_name"]] = tmp

        # for d in os.listdir(imgdir):
        #     print(name_to_id[d]["id"])
        self.dir = [os.path.join(imgdir,d) for d in os.listdir(imgdir)]

    
    def __len__(self):
        return len(self.dir)

    def __getitem__(self, idx):

        dir = self.dir[idx]
        img_path = os.path.join(dir, "image.tif")
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        class_file = sorted(glob.glob(os.path.join(dir,"class*.tif")))

        masks = []
        boxes = []

        for cf in class_file:
            mask = cv2.imread(cf, cv2.IMREAD_UNCHANGED)
            
            ids = np.unique(mask)
            for id in ids:
                if id == 0:
                    continue
                binary_mask = mask == id
                masks.append(torch.as_tensor(binary_mask, dtype=torch.uint8))

        masks = torch.stack(masks)
        for m in masks:
            pos = torch.where(m)
            xmin = torch.min(pos[1])
            xmax = torch.max(pos[1])
            ymin = torch.min(pos[0])
            ymax = torch.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(masks),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(masks),), dtype=torch.int64),
        }

        if self.transform:
            img, target = self.transform(img, target)

        return img, target


class ValDatasets(Dataset):

    def __init__(self, imgdir, jsondir, transform=None):
        self.imgdir = imgdir
        self.transform = transform
        self.coco = COCO(jsondir)
        self.image_id = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, idx):

        image_id = self.image_id[idx]
        annotation_id = self.coco.getAnnIds(image_id)
        annotation = self.coco.loadAnns(annotation_id)

        img_path = os.path.join(self.imgdir, str(image_id) + ".png")
        image = Image.open(img_path)
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        boxes = []
        labels = []

        for ann in annotation:
            xmin, ymin, w, h = ann['bbox']
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes)
        labels = torch.as_tensor(labels)
        image_id = torch.as_tensor(image_id)

        target = {
            "boxes": boxes,
            "labels": labels,
            "id": image_id
        }

        return image, target


class TestDatasets(Dataset):

    def __init__(self, imgdir, transform=None):
        self.data = []
        self.name = []
        self.transform = transform

        for img in os.listdir(imgdir):
            self.data.append(os.path.join(imgdir, img))
            self.name.append(int(img.split(".")[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        path = self.data[idx]
        img = Image.open(path)

        if self.transform:
            img = self.transform(img)

        return img, self.name[idx]


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def get_train_val_dataloader(imgdir, jsondir, transform=None,
                         batch_size=1, shuffle=False):
    """Get train dataloader"""
    train_dataset = TrainDatasets(imgdir, jsondir, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=4,
                                  pin_memory=True, collate_fn=collate_fn)
    return train_dataloader


def get_val_dataloader(imgdir, jsondir, transform=None,
                       batch_size=1, shuffle=False):
    """Get val dataloader"""
    val_dataset = ValDatasets(imgdir, jsondir, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=4,
                                collate_fn=collate_fn)
    return val_dataloader


def get_test_dataloader(imgdir, transform=None,
                        batch_size=1, shuffle=False):
    """Get test dataloader"""
    test_dataset = TestDatasets(imgdir, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=4)
    return test_dataloader

if __name__ == "__main__":
    dataset = TrainDatasets("data/train","data/test_image_name_to_ids.json")
    show_sample(dataset, index=0)