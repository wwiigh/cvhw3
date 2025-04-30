import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import numpy as np


from model import get_model
from dataset import get_train_val_dataloader


def evaluate(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model().to(device)

    model.load_state_dict(torch.load(path)['model_state_dict'])
    model.eval()
    model.model.roi_heads.detections_per_img = 1000

    train_dir = "data/train"
    _, val_dataloader = get_train_val_dataloader(train_dir)

    result = []

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

                for j in range(len(masks)):
                    mask = masks[j, 0].cpu().numpy()
                    mask = (mask > 0.5).astype(np.uint8)

                    rle = maskUtils.encode(np.asfortranarray(mask))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    if float(scores[j].item()) < 0.5:
                        continue
                    result.append({
                        "image_id": image_id,
                        "category_id": int(labels[j].item()),
                        "bbox": [boxes[j][0].item(), boxes[j][1].item(),
                                 boxes[j][2].item() - boxes[j][0].item(),
                                 boxes[j][3].item() - boxes[j][1].item()],
                        "segmentation": rle,
                        "score": float(scores[j].item())
                    })

    import json
    with open("segm_output.json", "w") as f:
        json.dump(result, f)
    coco_gt = COCO("annotations.json")
    coco_dt = coco_gt.loadRes(result)

    coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    cat_ids = coco_gt.getCatIds()  # [1~10]
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

    conf_matrix = np.zeros((len(cat_ids) + 1, len(cat_ids) + 1), dtype=int)

    for eval_img in coco_eval.evalImgs:
        if eval_img is None:
            continue

        gt_ids = eval_img['gtIds']
        dt_ids = eval_img['dtIds']
        dt_matches = eval_img['dtMatches'][0]
        gt_ignore = eval_img['gtIgnore']
        dt_ignore = eval_img['dtIgnore'][0]
        if (eval_img['aRng'] != [0, 1e5**2]):
            continue
        for i, dt_id in enumerate(dt_ids):
            if dt_ignore[i]:
                continue

            matched_gt_id = dt_matches[i]
            dt_ann = coco_dt.anns.get(dt_id)
            dt_cat = dt_ann['category_id']
            dt_idx = cat_id_to_idx.get(dt_cat, -1)

            if matched_gt_id == 0:
                conf_matrix[-1, dt_idx] += 1
            else:
                gt_ann = coco_gt.anns.get(matched_gt_id)
                gt_cat = gt_ann['category_id']
                gt_idx = cat_id_to_idx[gt_cat]
                conf_matrix[gt_idx, dt_idx] += 1

        matched_gt_ids = set(dt_matches[dt_matches > 0])
        for gt_id, ignore in zip(gt_ids, gt_ignore):
            if ignore:
                continue
            if gt_id not in matched_gt_ids:
                gt_ann = coco_gt.anns.get(gt_id)
                gt_cat = gt_ann['category_id']
                gt_idx = cat_id_to_idx[gt_cat]
                conf_matrix[gt_idx, -1] += 1

    labels = [coco_gt.loadCats([i])[0]['name'] for i in cat_ids]
    labels += ['background']

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels,
                cmap='Blues')

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (with background)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    evaluate("model/maskrcnn_50/exp2/exp2_19_final.pth")
