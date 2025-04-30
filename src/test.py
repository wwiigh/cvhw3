import torch
from tqdm import tqdm
from pycocotools import mask as maskUtils
import numpy as np


from model import get_model
from dataset import get_test_dataloader


def test(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model().to(device)
    model.load_state_dict(torch.load(path)['model_state_dict'])
    print(sum(p.numel() for p in model.parameters()))

    model.eval()
    model.model.roi_heads.detections_per_img = 1000

    test_dir = "data/test_release"
    test_json = "data/test_image_name_to_ids.json"
    test_dataloader = get_test_dataloader(test_dir, test_json)

    result = []

    with torch.no_grad():
        for images, id in tqdm(test_dataloader):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                image_id = id[0]
                masks = output["masks"]
                labels = output["labels"]
                scores = output["scores"]
                boxes = output["boxes"]

                for j in range(len(masks)):
                    mask = masks[j, 0].cpu().numpy()
                    mask = (mask > 0.5).astype(np.uint8)

                    rle = maskUtils.encode(np.asfortranarray(mask))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    if float(scores[j].item()) < 0.5:
                        pass
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
    with open("test-results.json", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":

    test("model/seblock_train5/exp5_49_final.pth")
