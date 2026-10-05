import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)


class AppearanceEncoder:
    """
    Fixed Phase 1H-0 appearance baseline.

    ImageNet-pretrained ResNet-50 with classifier removed.

    Vehicle crops are padded to square before resizing so that
    elongated vehicle boxes are not truncated by ImageNet's
    ordinary center-crop preprocessing.

    Output:
        2048-D L2-normalized appearance embedding.
    """

    def __init__(self, device=None):

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        weights = ResNet50_Weights.IMAGENET1K_V2

        model = resnet50(weights=weights)

        self.backbone = nn.Sequential(
            *list(model.children())[:-1]
        )

        self.backbone.eval()
        self.backbone.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.embedding_dimension = 2048

    @staticmethod
    def pad_to_square(image):

        width, height = image.size

        size = max(width, height)

        left = (size - width) // 2
        right = size - width - left

        top = (size - height) // 2
        bottom = size - height - top

        return ImageOps.expand(
            image,
            border=(left, top, right, bottom),
            fill=(0, 0, 0),
        )

    @torch.no_grad()
    def encode(self, image):

        if not isinstance(image, Image.Image):
            raise TypeError(
                "encode() expects a PIL.Image.Image"
            )

        image = image.convert("RGB")

        image = self.pad_to_square(image)

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        features = self.backbone(tensor)

        features = torch.flatten(
            features,
            start_dim=1,
        )

        features = torch.nn.functional.normalize(
            features,
            p=2,
            dim=1,
        )

        return features[0].cpu()
