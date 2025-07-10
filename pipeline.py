import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt


# --- Helper Functions ---
def tile_image(image, tile_size=512, stride=384):
    H, W = image.shape[:2]
    tiles = []
    positions = []
    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            tiles.append(image[y:y+tile_size, x:x+tile_size])
            positions.append((x, y))
    return tiles, positions

def run_yolo_on_tiles(tiles, positions, model):
    global_boxes = []
    for tile, (x_offset, y_offset) in zip(tiles, positions):
        results = model.predict(tile, imgsz=512, conf=0.4, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        for x1, y1, x2, y2 in boxes:
            global_boxes.append((x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset))
    return global_boxes

def iou(boxA, boxB):
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    areaA = (xa2 - xa1) * (ya2 - ya1)
    areaB = (xb2 - xb1) * (yb2 - yb1)
    return inter / (areaA + areaB - inter + 1e-6)

def merge_boxes(boxes, iou_thresh=0.5):
    boxes = boxes.copy()
    merged = []
    while boxes:
        base = boxes.pop(0)
        to_merge = [base]
        boxes_new = []
        for b in boxes:
            if iou(base, b) > iou_thresh:
                to_merge.append(b)
            else:
                boxes_new.append(b)
        x1 = min([b[0] for b in to_merge])
        y1 = min([b[1] for b in to_merge])
        x2 = max([b[2] for b in to_merge])
        y2 = max([b[3] for b in to_merge])
        merged.append((x1, y1, x2, y2))
        boxes = boxes_new
    return merged

def draw_boxes_on_image(image, boxes):
    img = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{i}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img

def estimate_fill_from_crop(crop, tank_id, out_dir):
    h, w = crop.shape[:2]
    size = min(h, w)
    crop_resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop_resized, cv2.COLOR_RGB2GRAY)
    
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = int(size * 0.45)
    cv2.circle(mask, center, radius, 255, -1)  # type: ignore

    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    _, shadow_mask = cv2.threshold(gray_masked, 80, 255, cv2.THRESH_BINARY_INV)

    shadow_pixels = np.sum((shadow_mask > 0) & (mask > 0))
    total_pixels = np.sum(mask > 0)
    shadow_pct = (shadow_pixels / total_pixels) * 100
    fill_pct = 100 - shadow_pct

    # Save intermediate visualizations
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(f"{out_dir}/tank_{tank_id}_crop.png", cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{out_dir}/tank_{tank_id}_mask.png", mask)
    cv2.imwrite(f"{out_dir}/tank_{tank_id}_shadow.png", shadow_mask)

    return round(fill_pct, 2)


def process_image(image_path, model_path="weights/best.pt"):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # type: ignore

    tiles, positions = tile_image(image_rgb)
    global_boxes = run_yolo_on_tiles(tiles, positions, model)
    merged_boxes = merge_boxes(global_boxes)

    # Save merged image with boxes
    merged_img = draw_boxes_on_image(image_rgb, merged_boxes)
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/image_with_boxes.jpg", cv2.cvtColor(merged_img, cv2.COLOR_RGB2BGR))

    results = []
    for i, (x1, y1, x2, y2) in enumerate(merged_boxes):
        pad = 5
        x1 = max(x1 - pad, 0)
        y1 = max(y1 - pad, 0)
        x2 = min(x2 + pad, image.shape[1]) # type: ignore
        y2 = min(y2 + pad, image.shape[0]) # type: ignore
        crop = image_rgb[y1:y2, x1:x2]
        fill = estimate_fill_from_crop(crop, i, "output/intermediate")
        results.append({"tank_id": i, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "fill_percent": fill})

    df = pd.DataFrame(results)
    df.to_csv("output/fill_estimates.csv", index=False)
    return df
