import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from PIL import Image


def pil2cv(image):
    """PIL型 -> OpenCV型"""
    if isinstance(image, str):  # Check if the input is a file pat
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
    """OpenCV型 -> PIL型"""
    if isinstance(image, str):  # Check if the input is a file pat
        image = Image.open(image)  # Load the image using PIL
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    return new_image


def resize_image(image, max_size=1024):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image


def get_random_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [cnt for cnt in contours if 100 < cv2.contourArea(cnt) < 10000]
    if not candidates:
        return None
    return random.choice(candidates)


def apply_color_change(image, contour):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    delta = random.randint(-15, 15)
    hsv[..., 0] = (hsv[..., 0] + delta) % 180
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    center = get_contour_center(contour)
    if 0 <= center[0] < image.shape[1] and 0 <= center[1] < image.shape[0]:
        blended = cv2.seamlessClone(
            result, image, mask, center, cv2.NORMAL_CLONE
        )
        return blended
    else:
        return image  # Return the original image if the center is out of bounds


def get_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return (0, 0)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def move_object(image, contour):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    bbox = cv2.boundingRect(contour)
    x, y, w, h = bbox
    obj_roi = image[y : y + h, x : x + w]
    mask_roi = mask[y : y + h, x : x + w]

    tx = random.randint(-50, 50)
    ty = random.randint(-50, 50)

    dst = image.copy()
    center = (x + tx + w // 2, y + ty + h // 2)
    dst = cv2.seamlessClone(obj_roi, dst, mask_roi, center, cv2.NORMAL_CLONE)
    return dst


def delete_object(image, contour):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return inpainted


def scale_object(image, contour, scale_factor=1.2):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    bbox = cv2.boundingRect(contour)
    x, y, w, h = bbox
    obj = image[y : y + h, x : x + w]
    mask_roi = mask[y : y + h, x : x + w]

    new_size = (int(w * scale_factor), int(h * scale_factor))
    resized_obj = cv2.resize(obj, new_size)
    resized_mask = cv2.resize(mask_roi, new_size)

    dst = image.copy()
    center = (x + w // 2, y + h // 2)
    dst = cv2.seamlessClone(resized_obj, dst, resized_mask, center, cv2.NORMAL_CLONE)
    return dst

def save_side_by_side(original, modified):
    height = max(original.shape[0], modified.shape[0])
    width = original.shape[1] + modified.shape[1]
    result = np.zeros((height, width, 3), dtype=np.uint8)
    result[:original.shape[0], :original.shape[1]] = original
    result[:modified.shape[0], original.shape[1]:] = modified
    return result

def generate_answer_image(original, modified, threshold=30):
    diff = cv2.absdiff(original, modified)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # 輪郭検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    answer = modified.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(answer, (x, y), (x + w, y + h), (0, 0, 255), 3)
    return answer

def generate_difference_pair(image, num_differences):
    image = resize_image(image)
    modified = image.copy()

    for _ in range(num_differences):  # Apply 5 random changes
        contour = get_random_contour(modified)
        if contour is None:
            continue
        operation = random.choice(["color", "move", "delete", "scale"])
        if operation == "color":
            modified = apply_color_change(modified, contour)
        elif operation == "move":
            modified = move_object(modified, contour)
        elif operation == "delete":
            modified = delete_object(modified, contour)
        elif operation == "scale":
            modified = scale_object(
                modified, contour, scale_factor=random.uniform(0.7, 1.3)
            )
    t = save_side_by_side(image, modified)
    answer = generate_answer_image(image, modified, threshold=30)
    return image, modified, t, answer
