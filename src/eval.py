import os
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from model import get_model
from dataset import get_train_val_dataloader
import numpy as np

def evaluate(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入模型
    model = get_model().to(device)
    model.load_state_dict(torch.load(path)['model_state_dict'])
    model.eval()

    # 載入 val 資料集
    train_dir = "data/train"
    _, val_dataloader = get_train_val_dataloader(train_dir)

    result = []
    anns = []

    with torch.no_grad():
        for images, targets in tqdm(val_dataloader):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                image_id = targets[i]["image_id"].item()
                masks = output["masks"]  # [N, 1, H, W]
                labels = output["labels"]
                scores = output["scores"]
                boxes = output["boxes"]

        # 格式化為 COCO 格式的結果
        
            
                for j in range(len(masks)):
                    mask = masks[j, 0].cpu().numpy()
                    mask = (mask > 0.5).astype(np.uint8)  # 二值化

                    # RLE encode
                    rle = maskUtils.encode(np.asfortranarray(mask))
                    rle["counts"] = rle["counts"].decode("utf-8")  # 轉成 str
                    if float(scores[j].item()) < 0.5:
                        continue
                    result.append({
                        "image_id": image_id,
                        "category_id": int(labels[j].item()),
                        "bbox": [boxes[j][0].item(), boxes[j][1].item(), boxes[j][2].item() - boxes[j][0].item(), boxes[j][3].item() - boxes[j][1].item()],
                        "segmentation": rle,
                        "score": float(scores[j].item())
                    })


                # 如果你要計算 mAP，記得要也準備 gt anns
                # anns.append(targets[i])


    # 可以選擇把結果存下來
    import json
    with open("segm_output.json", "w") as f:
        json.dump(result, f)
    # ========== Eval ==========
    # coco = COCO()  # 初始化空的 COCO
    coco_gt = COCO("annotations.json")
    coco_dt = coco_gt.loadRes(result)

    coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    


if __name__ == "__main__":
    # print("here")

    evaluate("model/exp1/exp1_29_final.pth")