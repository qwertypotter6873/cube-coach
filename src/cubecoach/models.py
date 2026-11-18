"""큐브 한 면(3x3)의 색을 읽는 모델들.

- hsv_rule      : 학습 없는 참고선. 칸 중심의 평균색을 HSV 임계값으로 나눈다.
- color_features: 고전 ML 베이스라인(kNN 등)의 입력. 칸 중심 평균색의 Lab + HSV.
- StickerCNN    : 베이스라인 CNN. 칸 하나(32x32)만 보고 색을 맞힌다. 주변 맥락이 없다.
- FaceGridNet   : 제안 모델. 면 전체(96x96)를 보고 3x3 칸의 색을 한 번에 낸다.
- mobilenet_grid: 전이학습 비교군. ImageNet 사전학습 MobileNetV3-Small 백본.

모든 신경망은 0~1 범위 RGB (B, 3, H, W)를 받고, 정규화는 모델 안에서 한다.
브라우저에서는 캔버스 픽셀을 255로 나누기만 하면 된다.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

from .synth import FACE_FILL, cell_centers

N_COLORS = 6


# --- 고전 방식 ------------------------------------------------------------

def _cell_means(img: np.ndarray, frac: float = 0.45) -> np.ndarray:
    """정렬이 맞았다고 가정한 9칸 중심 패치의 평균 RGB (9, 3), 0~1."""
    size = img.shape[0]
    half = int(FACE_FILL / 3 * size * frac / 2)
    out = np.empty((9, 3), dtype=np.float32)
    for k, (x, y) in enumerate(cell_centers(size).astype(int)):
        out[k] = img[y - half : y + half + 1, x - half : x + half + 1].reshape(-1, 3).mean(0)
    return out / 255.0


def color_features(img: np.ndarray) -> np.ndarray:
    """(9, 6): 칸별 평균색의 Lab과 HSV."""
    rgb = _cell_means(img)[None]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[0] / np.array([100, 128, 128], dtype=np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0] / np.array([360, 1, 1], dtype=np.float32)
    return np.concatenate([lab, hsv], axis=1)


def hsv_rule(img: np.ndarray) -> np.ndarray:
    """(9,) 색 번호. COLORS = "WRGYOB"."""
    hsv = cv2.cvtColor(_cell_means(img)[None], cv2.COLOR_RGB2HSV)[0]
    out = np.empty(9, dtype=np.int64)
    for k, (h, s, v) in enumerate(hsv):
        if s < 0.25 and v > 0.45:
            out[k] = 0  # W
        elif h < 12 or h >= 330:
            out[k] = 1  # R
        elif h < 38:
            out[k] = 4  # O
        elif h < 75:
            out[k] = 3  # Y
        elif h < 180:
            out[k] = 2  # G
        else:
            out[k] = 5  # B
    return out


def cell_patches(imgs: np.ndarray, out: int = 32) -> np.ndarray:
    """(N, S, S, 3) → (N*9, out, out, 3). StickerCNN 입력용으로 칸 자리를 잘라낸다."""
    size = imgs.shape[1]
    half = FACE_FILL / 3 * size / 2
    res = np.empty((len(imgs) * 9, out, out, 3), dtype=np.uint8)
    boxes = [(int(round(x - half)), int(round(y - half))) for x, y in cell_centers(size)]
    side = int(round(2 * half))
    for i, img in enumerate(imgs):
        for k, (x0, y0) in enumerate(boxes):
            x0, y0 = max(x0, 0), max(y0, 0)
            res[i * 9 + k] = cv2.resize(img[y0 : y0 + side, x0 : x0 + side], (out, out), interpolation=cv2.INTER_AREA)
    return res


# --- 신경망 ---------------------------------------------------------------

class _Norm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - 0.5) / 0.25


def _conv(i: int, o: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


def _dsconv(i: int, o: int, stride: int) -> nn.Sequential:
    """깊이별 분리 합성곱: 같은 수용 영역을 훨씬 적은 가중치로."""
    return nn.Sequential(
        nn.Conv2d(i, i, 3, stride, 1, groups=i, bias=False), nn.BatchNorm2d(i), nn.ReLU(inplace=True),
        nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class StickerCNN(nn.Module):
    """칸 하나 (B, 3, 32, 32) → (B, 6)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _Norm(),
            _conv(3, 16, 1), nn.MaxPool2d(2),
            _conv(16, 32, 1), nn.MaxPool2d(2),
            _conv(32, 64, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, N_COLORS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FaceGridNet(nn.Module):
    """면 전체 (B, 3, 96, 96) → (B, 6, 3, 3).

    출력의 (행, 열)이 곧 스티커 자리다. 각 자리의 판단에 면 전체의 평균 특징을
    이어 붙여서, "이 조명에서 흰색이 어떻게 보이는가" 같은 맥락을 쓰게 한다.
    """

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _Norm(),
            _conv(3, w, 2),  # 48
            _dsconv(w, 2 * w, 2), _dsconv(2 * w, 2 * w, 1),  # 24
            _dsconv(2 * w, 4 * w, 2), _dsconv(4 * w, 4 * w, 1),  # 12
            _dsconv(4 * w, 6 * w, 2), _dsconv(6 * w, 6 * w, 1),  # 6
            nn.AvgPool2d(2),  # 3
        )
        self.head = nn.Sequential(nn.Conv2d(12 * w, 6 * w, 1), nn.ReLU(inplace=True), nn.Conv2d(6 * w, N_COLORS, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        context = f.mean(dim=(2, 3), keepdim=True).expand_as(f)
        return self.head(torch.cat([f, context], dim=1))


class _MobileNetGrid(nn.Module):
    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights).features  # 96 → 3, 576채널
        self.head = nn.Conv2d(576, N_COLORS, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone((x - self.mean) / self.std))


def mobilenet_grid(pretrained: bool = True) -> nn.Module:
    return _MobileNetGrid(pretrained)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
