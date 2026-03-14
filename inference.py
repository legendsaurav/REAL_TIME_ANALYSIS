import cv2
import torch
import numpy as np
from model import YoloModel
from config import *
from utils import decode_predictions, draw_boxes

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return torch.tensor(img, dtype=torch.float32)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YoloModel().to(device)
    model.load_state_dict(torch.load('checkpoints/yolo_epoch_50.pth', map_location=device))
    model.eval()

    img_path = 'test.jpg'  # Change to your test image
    img_tensor = preprocess_image(img_path).to(device)
    with torch.no_grad():
        predictions = model(img_tensor)
    boxes, labels, scores = decode_predictions(predictions.cpu().numpy()[0])

    img = cv2.imread(img_path)
    img = draw_boxes(img, boxes, labels, scores)
    cv2.imshow('YOLO Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
