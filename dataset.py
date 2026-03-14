import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import IMAGE_SIZE, GRID_SIZE, NUM_CLASSES, BBOXES_PER_CELL

class YoloDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace('.jpg', '.txt'))

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = self.transform(img)

        # Load label
        target = np.zeros((GRID_SIZE, GRID_SIZE, BBOXES_PER_CELL * 5 + NUM_CLASSES), dtype=np.float32)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    grid_x = int(x * GRID_SIZE)
                    grid_y = int(y * GRID_SIZE)
                    for b in range(BBOXES_PER_CELL):
                        if target[grid_y, grid_x, b*5+4] == 0:
                            target[grid_y, grid_x, b*5:b*5+5] = [x, y, w, h, 1]
                            target[grid_y, grid_x, BBOXES_PER_CELL*5 + int(class_id)] = 1
                            break
        return img, torch.tensor(target)
