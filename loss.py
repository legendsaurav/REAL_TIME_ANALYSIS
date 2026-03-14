import torch
import torch.nn as nn
from config import GRID_SIZE, BBOXES_PER_CELL, NUM_CLASSES

class YoloLoss(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        # predictions: [batch, S, S, B*5 + C]
        # targets: [batch, S, S, B*5 + C]
        obj_mask = targets[..., 4] > 0
        noobj_mask = targets[..., 4] == 0

        # Localization loss (x, y, w, h)
        loc_loss = self.mse(predictions[obj_mask][..., :4], targets[obj_mask][..., :4])

        # Confidence loss
        conf_loss_obj = self.bce(predictions[obj_mask][..., 4], targets[obj_mask][..., 4])
        conf_loss_noobj = self.bce(predictions[noobj_mask][..., 4], targets[noobj_mask][..., 4])

        # Classification loss
        class_loss = self.bce(predictions[obj_mask][..., 5:], targets[obj_mask][..., 5:])

        total_loss = (
            self.lambda_coord * loc_loss +
            conf_loss_obj +
            self.lambda_noobj * conf_loss_noobj +
            class_loss
        )
        return total_loss
