"""CNN backbones wrapped in one interface: features -> pooled embedding -> classifier.

The shared `features` block is what Grad-CAM hooks into, and `embed()` gives the
image feature vector that the Random Forest layer uses.
"""
import torch
import torch.nn as nn
from torchvision import models as tvm

ARCHS = ["resnet18", "resnet50", "densenet121", "efficientnet_b0", "mobilenet_v3_large"]


def _backbone(arch, pretrained):
    weights = "DEFAULT" if pretrained else None
    net = getattr(tvm, arch)(weights=weights)
    if arch.startswith("resnet"):
        return nn.Sequential(*list(net.children())[:-2]), nn.Identity()
    if arch.startswith("densenet"):  # densenet applies ReLU after its feature block
        return net.features, nn.ReLU()
    return net.features, nn.Identity()


class OmniNet(nn.Module):
    def __init__(self, arch, n_classes, pretrained=True, dropout=0.3):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"Architecture must be one of {ARCHS}")
        self.arch = arch
        self.features, self.post = _backbone(arch, pretrained)
        with torch.no_grad():
            dim = self.post(self.features(torch.zeros(1, 3, 64, 64))).shape[1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes))
        self.embed_dim = dim

    def embed(self, x):
        return self.pool(self.post(self.features(x))).flatten(1)

    def forward(self, x):
        return self.head(self.embed(x))

    def freeze_backbone(self, frozen=True):
        for p in self.features.parameters():
            p.requires_grad = not frozen


def save_checkpoint(path, model, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = OmniNet(meta["architecture"], len(meta["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), meta
