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


def area_weights(n_in: int, n_out: int):
    """n_in 칸을 n_out 칸으로 줄일 때 칸마다 겹치는 길이 (n_out x n_in, 행 합 = n_in / n_out). 브라우저판 seg.ts 와 같은 계산."""
    import numpy as np

    s = n_in / n_out
    w = np.zeros((n_out, n_in), dtype=np.float64)
    for i in range(n_out):
        a, b = i * s, (i + 1) * s
        for k in range(int(a), min(n_in, int(np.ceil(b)))):
            w[i, k] = max(0.0, min(b, k + 1) - max(a, k))
    return w


def maps_to(logp, heat, n_out: int = 48):
    """모델 지도(로그확률 C x n x n, 중심 지도 n x n)를 n_out x n_out 으로 넓이 평균해 줄인다 (확률로 평균한 뒤 다시 로그).
    입력 256 실험 갈래에서 추적기가 늘 보던 48x48 지도를 받게 했다(README 5.2절, 지금 앱은 쓰지 않음)."""
    import numpy as np

    n = heat.shape[0]
    if n == n_out:
        return logp, heat
    w = area_weights(n, n_out)
    norm = (n / n_out) ** 2
    p = np.exp(logp.astype(np.float64))
    p2 = np.einsum("ik,ckl,jl->cij", w, p, w) / norm
    p2 /= p2.sum(0, keepdims=True)
    h2 = (w @ heat.astype(np.float64) @ w.T) / norm
    return np.log(np.maximum(p2, 1e-12)).astype(np.float32), h2.astype(np.float32)


APP_SIZES = (192, 256)  # export_onnx 가 확인하는 입력 한 변: 지금 앱은 192, 입력 256 실험 갈래(커밋 5154035)는 추적 단계에 256


def export_onnx(model, path, sizes=APP_SIZES) -> float:
    """입력 높이·너비를 열어 둔 ONNX 를 내보내고, 크기마다(192·256) torch 와 같은지 본다. 최대 차이를 돌려준다.
    학습 스크립트(scripts/train_seg.py)와 실험용 내보내기(scripts/export_app_onnx.py)가 함께 쓴다. 192 입력이면 앱에서 그대로 쓸 수 있다."""
    import numpy as np
    import onnxruntime as ort

    model = model.cpu().eval()
    torch.onnx.export(model, (torch.rand(1, 3, sizes[0], sizes[0]),), str(path), input_names=["image"], output_names=["maps"],
                      opset_version=17, dynamo=False, dynamic_axes={"image": {2: "h", 3: "w"}, "maps": {2: "mh", 3: "mw"}})
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    with torch.no_grad():
        for n in sizes:
            x = torch.rand(1, 3, n, n)
            worst = max(worst, float(np.abs(sess.run(None, {"image": x.numpy()})[0] - model(x).numpy()).max()))
    if worst > 1e-3:
        raise RuntimeError(f"ONNX 출력이 torch 와 다르다 (최대 차이 {worst})")
    return worst


def load_segnet(path) -> "CubeSegNet":
    """저장한 가중치에서 폭을 알아내(첫 합성곱의 출력 채널 수) CubeSegNet 을 만들고 불러온다 (평가 모드)."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m = CubeSegNet(width=int(sd["e1.0.weight"].shape[0]))
    m.load_state_dict(sd)
    return m.eval()


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
