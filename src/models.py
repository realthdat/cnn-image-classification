import torch.nn as nn
import torchvision.models as models

class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes))
    def forward(self, x): return self.head(self.features(x))

def create_mobilenet(num_classes, freeze=True, pretrained=True):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
    if freeze:
        for p in m.features.parameters(): p.requires_grad = False
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m

def create_resnet18(num_classes, freeze=False, pretrained=True):
    r = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if freeze:
        for n,p in r.named_parameters():
            if not n.startswith("fc."): p.requires_grad = False
    r.fc = nn.Linear(r.fc.in_features, num_classes)
    return r
