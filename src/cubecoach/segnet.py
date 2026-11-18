"""연속 추적용 인식 모델: 화면에서 보이는 스티커의 색과 위치를 칸 단위로 낸다.

입력  (B, 3, 192, 192)  0~1 RGB
출력  (B, 9, 48, 48)    채널 0~7 = 분류 점수 (스티커 색 6 + 큐브 몸체 + 그 밖),
                        채널 8   = 스티커 중심 지도 (sigmoid 전)

- CubeSegNet: 제안 모델. 넓은 범위를 보는 인코더-디코더라서 조명·손·주변 스티커
  맥락을 쓴다.
- PixelMLP  : 베이스라인. 각 칸의 평균색만 보고 판단한다 (공간 맥락 없음).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .models import _conv, _dsconv, _Norm
from .render3d import N_CLASSES, STRIDE

OUT_CH = N_CLASSES + 1


class CubeSegNet(nn.Module):
    def __init__(self, width: int = 24) -> None:
        super().__init__()
        w = width
        self.norm = _Norm()
        self.e1 = _conv(3, w, 2)  # 1/2
        self.e2 = nn.Sequential(_dsconv(w, 2 * w, 2), _dsconv(2 * w, 2 * w, 1))  # 1/4
        self.e3 = nn.Sequential(_dsconv(2 * w, 4 * w, 2), _dsconv(4 * w, 4 * w, 1), _dsconv(4 * w, 4 * w, 1))  # 1/8
        self.e4 = nn.Sequential(_dsconv(4 * w, 8 * w, 2), _dsconv(8 * w, 8 * w, 1), _dsconv(8 * w, 8 * w, 1))  # 1/16
        self.d3 = _dsconv(12 * w, 4 * w, 1)
        self.d2 = _dsconv(6 * w, 2 * w, 1)
        self.head = nn.Conv2d(2 * w, OUT_CH, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.e2(self.e1(self.norm(x)))
        c = self.e3(b)
        d = self.e4(c)
        u = self.d3(torch.cat([F.interpolate(d, scale_factor=2.0, mode="nearest"), c], dim=1))
        u = self.d2(torch.cat([F.interpolate(u, scale_factor=2.0, mode="nearest"), b], dim=1))
        return self.head(u)


class PixelMLP(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _Norm(), nn.AvgPool2d(STRIDE),
            nn.Conv2d(3, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, OUT_CH, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


CLASS_WEIGHTS = torch.tensor([2.0] * 6 + [1.0, 0.5])  # 스티커 칸은 드물고 중요하다


def seg_loss(out: torch.Tensor, seg: torch.Tensor, heat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """분류 손실과 중심 지도 손실."""
    ce = F.cross_entropy(out[:, :N_CLASSES], seg, weight=CLASS_WEIGHTS.to(out.device))
    bce = F.binary_cross_entropy_with_logits(out[:, N_CLASSES], heat, weight=1 + 8 * heat)
    return ce, bce
