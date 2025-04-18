import os

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from torch.cuda.amp import autocast, GradScaler

from dataset import get_train_val_dataloader
# from utils import transform_val
from utils import get_transform
from model import get_model

def train():
    """Start training"""
    exp_dir = "exp1"
    if not os.path.exists(f"model/{exp_dir}"):
        os.makedirs(f"model/{exp_dir}")

    writer = SummaryWriter(f"logs/{exp_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    epochs = 10
    batch_size = 1
    learning_rate = 1e-4
    weight_decay = 1e-4
    momentum = 0.9
    T_max = 50
    train_dir = "data/train"
   
    
    
    transform = get_transform(True)
    train_dataloader, val_dataloader = get_train_val_dataloader(train_dir)
    model = get_model().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                            momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    scaler = torch.amp.GradScaler("cuda")
    for epoch in range(epochs):

        running_loss = 0

        model.train()
        for (image, target) in tqdm(train_dataloader,
                                    desc=f"Epoch {epoch+1}/{epochs}"):

            image = [img.to(device) for img in image]
            target = [{key: value.to(device) for key, value in t.items()}
                      for t in target]
            
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                output = model(image, target)
                loss = sum(loss for loss in output.values())

            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            # break

        print(f"Epoch [{epoch+1}/{epochs}], Loss: \
              {running_loss/(len(train_dataloader)):.4f}")
        writer.add_scalar("Loss/epoch", running_loss/(len(train_dataloader)),
                          epoch)

        # model.eval()
        # result = []
        # ans = []
        # with torch.no_grad():
        #     for (image, target) in tqdm(val_dataloader, desc="val"):
        #         image = [img.to(device) for img in image]
        #         ans.append(target)

        #         output = model(image)
        #         for i, out in enumerate(output):
        #             image_id = target[0]["image_id"].item()

        #             for index in range(len(out["boxes"])):
        #                 boxes = out["boxes"][index]
        #                 labels = out["labels"][index].item()
        #                 scores = out["scores"][index].item()

        #                 result.append({
        #                     "image_id": image_id,
        #                     "category_id": labels,
        #                     "bbox": [boxes[0].item(), boxes[1].item(),
        #                              boxes[2].item() - boxes[0].item(),
        #                              boxes[3].item() - boxes[1].item()],
        #                     "score": scores
        #                 })

        # use in https://zhuanlan.zhihu.com/p/134229574
        # coco_dt = coco.loadRes(result)
        # coco_gt = coco.loadRes(ans)
        # coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType="bbox")
        # coco_eval.evaluate()
        # coco_eval.accumulate()
        # coco_eval.summarize()

        # mAP50 = coco_eval.stats[1]
        # mAP75 = coco_eval.stats[2]
        # mAP_all = coco_eval.stats[0]
        # writer.add_scalar("mAP50", mAP50, epoch)
        # writer.add_scalar("mAP75", mAP75, epoch)
        # writer.add_scalar("mAP_all", mAP_all, epoch)

        # if best_map < mAP_all:
        #     best_map = mAP_all
        #     torch.save(
        #         {
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'scheduler_state_dict': scheduler.state_dict()
        #         },
        #         f"model/{exp_dir}/{exp_dir}_{epoch}_final.pth"
        #     )
        # else:
        #     torch.save(
        #         {
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'scheduler_state_dict': scheduler.state_dict()
        #         },
        #         f"model/{exp_dir}/{exp_dir}_{epoch}_final.pth"
        #     )
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
