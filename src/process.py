import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from PIL import Image

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    if isinstance(image, str):  # Check if the input is a file path
        image = Image.open(image)  # Load the image using PIL
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    if isinstance(image, str):  # Check if the input is a file path
        image = Image.open(image)  # Load the image using PIL
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    return new_image

def get_proposed_regions(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()  # 高速版
    rects = ss.process()
    regions = []
    for x, y, w, h in rects[:200]:  # 上位100個のみ使用
        if w * h > 500 and w > 20 and h > 20:
            regions.append((x, y, w, h))
    return regions

def adjust_color(image, delta_hue=10, delta_saturation=10, delta_value=10):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.int16)  # Convert to int16 to avoid overflow
    hsv[:, :, 0] = (hsv[:, :, 0] + delta_hue) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + delta_saturation, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + delta_value, 0, 255)
    hsv = hsv.astype(np.uint8)  # Convert back to uint8
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_natural_change(image, region):
    x, y, w, h = region
    modified = image.copy()
    # op = random.choice(["resize", "move", "color"])
    op = "color"  # For testing, we can fix the operation to "resize"

    if op == "resize":
        scale = random.uniform(0.5, 1.5)
        nw, nh = int(w * scale), int(h * scale)
        roi = image[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (nw, nh))

        # Clip dimensions to fit within the image boundaries
        nx = min(modified.shape[1] - nw, x)
        ny = min(modified.shape[0] - nh, y)
        nw = min(nw, modified.shape[1] - nx)
        nh = min(nh, modified.shape[0] - ny)

        modified[y:y+h, x:x+w] = 255
        modified[ny:ny+nh, nx:nx+nw] = roi_resized[:nh, :nw]

    elif op == "move":
        dx, dy = random.randint(-20, 20), random.randint(-20, 20)
        nx, ny = x + dx, y + dy
        nx = max(0, min(modified.shape[1] - w, nx))
        ny = max(0, min(modified.shape[0] - h, ny))
        roi = image[y:y+h, x:x+w]
        modified[y:y+h, x:x+w] = 255
        modified[ny:ny+h, nx:nx+w] = roi

    elif op == "color":
        delta_hue = random.randint(-10, 10)
        delta_saturation = random.randint(-30, 30)
        delta_value = random.randint(-30, 30)
        roi = image[y:y+h, x:x+w]
        roi_colored = adjust_color(roi, delta_hue, delta_saturation, delta_value)
        modified[y:y+h, x:x+w] = roi_colored

    return modified

def generate_spot_diff_large(image, num_changes=10):
    if isinstance(image, str):  # Check if the input is a file path
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Failed to load image from path: {image}")
    h, w = image.shape[:2]
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        image = cv2.resize(image, (int(w*scale), int(h*scale)))

    modified = image.copy()
    regions = get_proposed_regions(image)
    selected = random.sample(regions, min(num_changes, len(regions)))

    for region in selected:
        modified = apply_natural_change(modified, region)

    combined = np.hstack((image, modified))
    return modified