from pathlib import Path
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils


def decode_maskobj(mask_obj):
    return mask_utils.decode(mask_obj)


def encode_mask(binary_mask):
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def read_maskfile(filepath):
    mask_array = imageio.imread(filepath)
    return mask_array


image_subdir = './data/test_release/'
image_subdir = Path(image_subdir)

image_path = image_subdir / 'c8cb7626-7423-4c1e-a81c-5ff25ea180b3.tif'

image = cv2.imread(str(image_path))
# mask = sio.imread(mask_path)

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(image)
plt.axis('off')

# plt.subplot(122)
# plt.imshow(mask)
# plt.axis('off')

# plt.tight_layout()
plt.show()

rle_mask = {"size": [446, 512],
            "counts": "][n3?^=2O1N2N1O10000O1000000001N1N3N3K5Kiih2"}

decoded_mask = decode_maskobj(rle_mask)
plt.figure(figsize=(4, 3))
plt.imshow(decoded_mask)
plt.show()
