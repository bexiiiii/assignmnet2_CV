import torch
import torch.nn as nn


class DeepGestureCNN(nn.Module):
    """Exactly the A1 model (4 conv blocks, 128x128 input)."""

    def __init__(self, num_classes=3):
        super().__init__()
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)]
        self.features = nn.Sequential(*block(3, 32), *block(32, 64),
                                      *block(64, 128), *block(128, 128))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(128 * 16, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


# DINOv2 ViT-S/14 with 4 register tokens ("Vision Transformers Need Registers",
# Darcet et al., ICLR 2024). Registers remove the high-norm artefact tokens of
# plain DINOv2, so attention maps are much cleaner for the explanation part.
DINO_NAME = "vit_small_patch14_reg4_dinov2.lvd142m"


class DinoGesture(nn.Module):
    """DINOv2-reg ViT-S/14 backbone + small head. Last `train_blocks` blocks are
    fine-tuned, the rest stays frozen (315 training images is too few to
    move all 22M weights)."""

    def __init__(self, num_classes=3, train_blocks=4, pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(DINO_NAME, pretrained=pretrained,
                                          num_classes=0, img_size=224)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in self.backbone.blocks[-train_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
        dim = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim, num_classes))

    def embed(self, x):
        return self.backbone(x)  # CLS token after final norm

    def forward(self, x):
        return self.head(self.embed(x))


def count_params(m, trainable=False):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable))
