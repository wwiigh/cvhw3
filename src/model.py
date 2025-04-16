import torch.nn as nn
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, training=True):
        super().__init__()
        self.model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights='DEFAULT')
        # self.model = models.detection.fasterrcnn_resnet50_fpn(
        #   weights='DEFAULT')
        self.training = training
        num_classes = 11
        self.model.roi_heads.box_predictor.cls_score = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )
        self.model.roi_heads.box_predictor.bbox_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes * 4)
        )

    def forward(self, x, target=None):
        if target is not None:
            output = self.model(x, target)
        else:
            output = self.model(x)
        return output


def get_model(training=True):
    """Return Model with faster rcnn"""
    return Model(training=training)
