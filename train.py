import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import YoloDataset
from model import YoloModel
from loss import YoloLoss
from config import *
import numpy as np

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = YoloDataset(
        images_dir='dataset/images',
        labels_dir='dataset/labels'
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = YoloModel().to(device)
    criterion = YoloLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for images, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)
            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(loader):.4f}")

        torch.save(model.state_dict(), f"checkpoints/yolo_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
