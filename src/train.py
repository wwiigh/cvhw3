import os

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from dataset import get_train_dataloader, get_val_dataloader
from utils import transform_val
from utils import transform
from model import get_model


def train():
    """Start training"""
    exp_dir = "exp9"
    if not os.path.exists(f"model/{exp_dir}"):
        os.makedirs(f"model/{exp_dir}")

    writer = SummaryWriter(f"logs/{exp_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    epochs = 23
    batch_size = 1
    learning_rate = 5e-4
    weight_decay = 5e-4
    momentum = 0.9

    train_dir = "data/train"
    train_json = "data/train.json"
    val_dir = "data/valid"
    val_json = "data/valid.json"

    train_dataloader = get_train_dataloader(train_dir, train_json,
                                            batch_size=batch_size,
                                            transform=transform,
                                            shuffle=True)
    val_dataloader = get_val_dataloader(val_dir, val_json,
                                        transform=transform_val,
                                        batch_size=batch_size,
                                        shuffle=False)

    model = get_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[3, 6, 9,
                                                                 12, 15, 18],
                                                     gamma=0.5)

    coco_gt = COCO("data/valid.json")

    best_map = 0
    for epoch in range(epochs):

        running_loss = 0

        model.train()
        for (image, target) in tqdm(train_dataloader,
                                    desc=f"Epoch {epoch+1}/{epochs}"):

            image = [img.to(device) for img in image]
            target = [{key: value.to(device) for key, value in t.items()}
                      for t in target]

            output = model(image, target)

            loss = output["loss_classifier"] + output["loss_box_reg"] + \
                output["loss_objectness"] + output["loss_rpn_box_reg"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: \
              {running_loss/(len(train_dataloader)):.4f}")
        writer.add_scalar("Loss/epoch", running_loss/(len(train_dataloader)),
                          epoch)

        model.eval()
        result = []
        with torch.no_grad():
            for (image, target) in tqdm(val_dataloader, desc="val"):
                image = [img.to(device) for img in image]

                output = model(image)
                for i, out in enumerate(output):
                    image_id = target[i]["id"].item()

                    for index in range(len(out["boxes"])):
                        boxes = out["boxes"][index]
                        labels = out["labels"][index].item()
                        scores = out["scores"][index].item()

                        result.append({
                            "image_id": image_id,
                            "category_id": labels,
                            "bbox": [boxes[0].item(), boxes[1].item(),
                                     boxes[2].item() - boxes[0].item(),
                                     boxes[3].item() - boxes[1].item()],
                            "score": scores
                        })

        # use in https://zhuanlan.zhihu.com/p/134229574
        coco_dt = coco_gt.loadRes(result)
        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        mAP50 = coco_eval.stats[1]
        mAP75 = coco_eval.stats[2]
        mAP_all = coco_eval.stats[0]
        writer.add_scalar("mAP50", mAP50, epoch)
        writer.add_scalar("mAP75", mAP75, epoch)
        writer.add_scalar("mAP_all", mAP_all, epoch)

        if best_map < mAP_all:
            best_map = mAP_all
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                },
                f"model/{exp_dir}/{exp_dir}_{epoch}_final.pth"
            )
        else:
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                },
                f"model/{exp_dir}/{exp_dir}_{epoch}_final.pth"
            )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr:.6f}")

        writer.add_scalar("Learning Rate", current_lr, epoch)

    writer.close()

    print("save model")
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        },
        f"model/{exp_dir}/{exp_dir}_{epoch}_final.pth"
    )


if __name__ == "__main__":
    train()
