import numpy as np
import cv2
from config import GRID_SIZE, IMAGE_SIZE, NUM_CLASSES, BBOXES_PER_CELL

def iou(box1, box2):
    # box: [x_center, y_center, w, h]
    x1_min = box1[0] - box1[2]/2
    y1_min = box1[1] - box1[3]/2
    x1_max = box1[0] + box1[2]/2
    y1_max = box1[1] + box1[3]/2

    x2_min = box2[0] - box2[2]/2
    y2_min = box2[1] - box2[3]/2
    x2_max = box2[0] + box2[2]/2
    y2_max = box2[1] + box2[3]/2

    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        idxs = idxs[1:]
        idxs = [i for i in idxs if iou(boxes[current], boxes[i]) < iou_threshold]
    return keep

def decode_predictions(pred, conf_threshold=0.5):
    boxes = []
    labels = []
    scores = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            for b in range(BBOXES_PER_CELL):
                offset = b * 5
                conf = pred[i, j, offset+4]
                if conf > conf_threshold:
                    x = pred[i, j, offset+0]
                    y = pred[i, j, offset+1]
                    w = pred[i, j, offset+2]
                    h = pred[i, j, offset+3]
                    class_probs = pred[i, j, BBOXES_PER_CELL*5:]
                    class_id = np.argmax(class_probs)
                    score = conf * class_probs[class_id]
                    # Convert to absolute coordinates
                    x_abs = int(x * IMAGE_SIZE)
                    y_abs = int(y * IMAGE_SIZE)
                    w_abs = int(w * IMAGE_SIZE)
                    h_abs = int(h * IMAGE_SIZE)
                    boxes.append([x_abs, y_abs, w_abs, h_abs])
                    labels.append(class_id)
                    scores.append(score)
    if len(boxes) == 0:
        return [], [], []
    keep = non_max_suppression(boxes, scores)
    boxes = [boxes[i] for i in keep]
    labels = [labels[i] for i in keep]
    scores = [scores[i] for i in keep]
    return boxes, labels, scores

def draw_boxes(img, boxes, labels, scores):
    for box, label, score in zip(boxes, labels, scores):
        x, y, w, h = box
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"Class {label} {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return img
