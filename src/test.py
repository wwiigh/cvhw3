import csv

from tqdm import tqdm
import torch
import json

from model import get_model
from dataset import get_test_dataloader
from utils import transform_val


def test(path):
    """Start predict testing ans"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)

    model.load_state_dict(torch.load(path)['model_state_dict'])
    print(sum(p.numel() for p in model.parameters()))
    model.eval()

    testdir = "data/test"
    test_dataloader = get_test_dataloader(testdir,
                                          transform=transform_val,
                                          batch_size=1,
                                          shuffle=False)

    file = open("pred.json", mode='w')
    cs = open("pred.csv", mode='w', newline="")
    writer = csv.writer(cs)
    writer.writerow(["image_id", "pred_label"])
    ans = []
    for (image, id) in tqdm(test_dataloader):
        image = image.to(device)
        with torch.no_grad():
            output = model(image)
        output = output[0]
        tmp = []
        for i in range(len(output["boxes"])):
            if (output['scores'][i].item() < 0.5):
                continue
            x_min, y_min, x_max, y_max = output['boxes'][i].tolist()
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            ans.append({
                "image_id": id.item(),
                "bbox": bbox,
                "score": output['scores'][i].item(),
                "category_id": output['labels'][i].item()
            })
            tmp.append({
                "image_id": id.item(),
                "x_min": x_min,
                "category_id": output['labels'][i].item()
            })
        if len(tmp) == 0:
            writer.writerow([id.item(), -1])
        else:
            tmp.sort(key=lambda x: x["x_min"])
            ans_str = 0
            for d in tmp:
                ans_str = ans_str * 10 + int(d["category_id"]) - 1
            writer.writerow([id.item(), ans_str])

    json.dump(ans, file, indent=4, ensure_ascii=False)
    file.close()
    cs.close()
    print("finish test")


if __name__ == "__main__":
    test("model/exp9/exp9_22_final.pth")
