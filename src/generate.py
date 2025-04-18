import os
import json
import numpy as np
from PIL import Image
from skimage import measure
import cv2
print("here")
import imageio.v2 as imageio
from pycocotools import mask as maskUtils

print("here")

def tif_to_coco(root_dir, output_json_path):

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}]
    }

    annotation_id = 1
    image_id = 1
    print("here")
    dir = ['data/train\\1ac7828d-31dd-450b-b7bb-4b47af993d5e', 'data/train\\d572da05-c8b3-4a1a-bc1b-7e5fa1296269', 'data/train\\7188e545-85c4-4811-806b-a093bf860ac6', 'data/train\\f7b5e795-dae2-4393-81f3-61b6ab0f83e1', 'data/train\\4ac5db7c-cfd9-43df-9509-defe87c8b7db', 'data/train\\bc0a0945-571a-4a80-9199-90e7afbb631d', 'data/train\\38ba8875-369e-4595-87b4-00540eefdf5f', 'data/train\\e0414615-6cc4-49a9-84c7-9f53dcc39d67', 'data/train\\08a61c57-1c28-4655-9a76-f9baf929ebc4', 'data/train\\f2ba4abb-3f03-4fac-9acb-4a2299597cfb', 'data/train\\a6211e46-edf8-46f1-8f57-3ea7f4c73467', 'data/train\\e3691953-56da-4756-b116-ef84a406672d', 'data/train\\ff299f9b-54a1-4a59-884b-f105b8e2708c', 'data/train\\9a27538c-6d97-4a76-8bfc-9ccca2704b75', 'data/train\\0e661f91-df55-4ea7-88ee-072e79c780a7', 'data/train\\63270aa7-f660-4470-ba30-981b7721efcd', 'data/train\\cb9f5610-bc44-4c85-b430-31b5083aeb68', 'data/train\\109dc0df-e6a2-47c0-bf96-df35f0f099b9', 'data/train\\d42d86ab-2678-44fc-aef5-ca5222497516', 'data/train\\792c2aff-8a9d-41dc-a3e6-af44c7226a1e', 'data/train\\687fd906-bfa8-47b1-bcfc-ad515d87910c', 'data/train\\4535ce98-810f-4ffb-954f-e02087d38375', 'data/train\\afe14eef-3a95-4ced-bf91-04a034a3372d', 'data/train\\8a810fb7-4c9b-4d55-aa0f-1616b91ad44c', 'data/train\\86b3bd13-d73d-4406-ba6b-130abe41b595', 'data/train\\abde0341-1fd2-45ca-9545-947c3041fb27', 'data/train\\16312b02-2e7b-4e28-804b-b91406c2cc59', 'data/train\\97a40a91-7b51-4b78-8742-1b9603086cbd', 'data/train\\10352a94-c1bd-419d-a2cf-7a4c486b61f8', 'data/train\\d6d92d0d-54cf-4b11-8988-d43e6569e069', 'data/train\\ed834742-aab1-49df-8577-94568f3a581b', 'data/train\\587974ad-2c88-4f76-99f8-af685284028d', 'data/train\\80a4fbfd-f85e-4ac0-86c4-6ec7c27ccaa8', 'data/train\\560b4275-43d6-47b7-92dd-392fd4b1be75', 'data/train\\81a25474-a5b6-48aa-a8eb-38f488577bf5', 'data/train\\a0919b53-b7b5-4217-a28b-beb03a6a9fb9', 'data/train\\dcfa9c6d-cc2e-4330-85f5-60a2cb3b8f68', 'data/train\\7a09aa24-d957-433e-979a-abfc486c1b93', 'data/train\\70e6b406-d863-4be6-ade6-c4e73607ba56', 'data/train\\c2e24ac1-168d-4dfb-a42c-be41f38ebb95', 'data/train\\f696ba5f-46ee-4fb1-9f6a-091750d16401', 'data/train\\c91ebeed-6954-43cd-b29e-981c627591ae']
    for d in dir:
        # print(subdir, dirs, files)
        # if "image.tif" not in files:
        #     continue

        image_path = os.path.join(d, "image.tif")
        img = Image.open(image_path)
        w, h = img.size

        coco["images"].append({
            "id": image_id,
            "file_name": os.path.relpath(image_path, root_dir).replace("\\", "/"),
            "height": h,
            "width": w
        })

        # collect all class_n.tif files
        for f in ["class1.tif", "class2.tif", "class3.tif", "class4.tif"]:
            p = os.path.join(d,f)
            if os.path.isfile(p):
                mask_path = p
                mask = imageio.imread(mask_path)
                mask_np = np.array(mask)

                # unique instance id (background = 0)
                instance_ids = np.unique(mask_np)
                instance_ids = instance_ids[instance_ids != 0]

                for inst_id in instance_ids:
                    binary_mask = (mask_np == inst_id).astype(np.uint8)
                    if binary_mask.sum() == 0:
                        continue

                    rle = maskUtils.encode(np.asfortranarray(binary_mask))
                    rle["counts"] = rle["counts"].decode("utf-8")

                    coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "segmentation": rle,
                        "iscrowd": 0,
                        "area": int(maskUtils.area(rle)),
                        "bbox": list(map(float, maskUtils.toBbox(rle))),
                    })
                    annotation_id += 1

        image_id += 1

    # save
    with open(output_json_path, "w") as f:
        json.dump(coco, f)
    print(f"✅ COCO annotation saved to: {output_json_path}")

if __name__ == "__main__":
    print("here")
    
    tif_to_coco("data/train", "data/train/annotations.json")