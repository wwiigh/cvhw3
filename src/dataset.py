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
from utils import transform

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

def limit_instances_by_area(masks, boxes, labels, max_instances=30):
    areas = [mask.sum().item() for mask in masks]
    sorted_indices = sorted(range(len(areas)), key=lambda i: -areas[i])
    selected = sorted_indices[:max_instances]
    return masks[selected], boxes[selected], labels[selected]

class TrainDatasets(Dataset):
    def __init__(self, imgdir, transform=None):
        # self.imgdir = imgdir
        self.transform = transform
        # json_file = open(json_path,'r')
        # f =  json_file.read()   # 要先使用 read 讀取檔案
        # name_to_id_tmp = json.loads(f) 
        # name_to_id = {}
        # for tmp in name_to_id_tmp:
        #     name_to_id[tmp["file_name"]] = tmp

        # for d in os.listdir(imgdir):
        #     print(name_to_id[d]["id"])
        self.dir = imgdir

    
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
            "image_id": torch.tensor(idx+1),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(masks),), dtype=torch.int64),
        }

        masks, boxes, labels = limit_instances_by_area(
        target["masks"], target["boxes"], target["labels"], max_instances=30
        )

        if self.transform:
            img, target = self.transform(img, target)


        return img, target


class TestDatasets(Dataset):

    def __init__(self, imgdir, json_path, transform=None):
        self.imgdir = imgdir
        self.transform = transform
        json_file = open(json_path,'r')
        f =  json_file.read()   # 要先使用 read 讀取檔案
        name_to_id_tmp = json.loads(f) 
        name_to_id = {}
        for tmp in name_to_id_tmp:
            name_to_id[tmp["file_name"]] = tmp

        self.file = []
        self.id = []
        for d in os.listdir(imgdir):
            self.file.append(d)
            self.id.append(name_to_id[d]["id"])
            # print(name_to_id[d]["id"])


    
    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):

        
        img_path = os.path.join(self.imgdir, self.file[idx])
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # class_file = sorted(glob.glob(os.path.join(dir,"class*.tif")))

        # masks = []
        # boxes = []

        # for cf in class_file:
        #     mask = cv2.imread(cf, cv2.IMREAD_UNCHANGED)
            
        #     ids = np.unique(mask)
        #     for id in ids:
        #         if id == 0:
        #             continue
        #         binary_mask = mask == id
        #         masks.append(torch.as_tensor(binary_mask, dtype=torch.uint8))

        # masks = torch.stack(masks)
        # for m in masks:
        #     pos = torch.where(m)
        #     xmin = torch.min(pos[1])
        #     xmax = torch.max(pos[1])
        #     ymin = torch.min(pos[0])
        #     ymax = torch.max(pos[0])
        #     boxes.append([xmin, ymin, xmax, ymax])
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # labels = torch.ones((len(masks),), dtype=torch.int64)

        # target = {
        #     "boxes": boxes,
        #     "labels": labels,
        #     "masks": masks,
        #     "image_id": torch.tensor(idx+1),
        #     "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
        #     "iscrowd": torch.zeros((len(masks),), dtype=torch.int64),
        # }

        # masks, boxes, labels = limit_instances_by_area(
        # target["masks"], target["boxes"], target["labels"], max_instances=30
        # )

        # if self.transform:
        #     img, target = self.transform(img, target)


        return img, self.id[idx]


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def collate_fn_test(batch):
    images = zip(*batch)
    return list(images)

def get_train_val_dataloader(imgdir,
                         batch_size=1, shuffle=False):
    """Get train dataloader"""
    all_dirs = sorted([
        os.path.join(imgdir, d) for d in os.listdir(imgdir)
    ])

    train_dirs, val_dirs = train_test_split(all_dirs, test_size=0.2, random_state=42)
    train_dataset = TrainDatasets(train_dirs, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=1,
                                  pin_memory=True, collate_fn=collate_fn)
    
    val_dataset = TrainDatasets(val_dirs, transform=None)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=1,
                                  pin_memory=True, collate_fn=collate_fn)

    return train_dataloader, val_dataloader


def get_test_dataloader(imgdir, jsonpath, transform=None,
                        batch_size=1, shuffle=False):
    """Get test dataloader"""
    test_dataset = TestDatasets(imgdir, jsonpath, transform=None)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=1,
                                  pin_memory=True, collate_fn=collate_fn_test)
    return test_dataloader

if __name__ == "__main__":
    # traind_loader, val_loader = get_train_val_dataloader("data/train")
    test_loader = get_test_dataloader("data/test_release", "data/test_image_name_to_ids.json")
    