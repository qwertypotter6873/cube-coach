"""README용 전처리 전·후 그림.

윗줄(실제 영상): 원본 프레임 → 추적기가 정한 관심영역만 잘라 192x192 → 모델 입력 (0~1로 나눈 뒤 (x−0.5)/0.25,
                  모델 안의 _Norm 과 같은 고정 계산. 선형이라 그대로 그리면 잘라 낸 그림과 같으므로 빨강 채널 값을 색으로 보인다)
아랫줄(합성 학습 데이터): 렌더 원본 → 정답 분류 지도(384→48, 칸마다 다수결) → 스티커 중심 지도(가우시안)
실행: .venv/bin/python scripts/fig_preprocess.py   결과: reports/preprocess.png
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cubecoach.render3d import render, sample_scene  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAL = np.array([[244, 244, 240], [214, 35, 58], [4, 168, 70], [233, 224, 44], [250, 106, 44], [4, 100, 188], [70, 70, 70], [0, 0, 0]], np.uint8)
plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]


def main() -> None:
    meta = json.loads((ROOT / "data" / "real" / "label_frames" / "frames.json").read_text())[12]
    cap = cv2.VideoCapture(str(ROOT / "data" / "raw" / meta["video"]))
    cap.set(cv2.CAP_PROP_POS_MSEC, meta["t"] * 1000)
    _, fr = cap.read()
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    H, W = fr.shape[:2]
    cx, cy, s = meta["roi"]
    side = s * W
    x0, y0 = int(W / 2 + cx * W - side / 2), int(H / 2 + cy * W - side / 2)
    boxed = fr.copy()
    cv2.rectangle(boxed, (x0, y0), (int(x0 + side), int(y0 + side)), (255, 255, 255), 6)
    crop = cv2.resize(fr[max(y0, 0):int(y0 + side), max(x0, 0):int(x0 + side)], (192, 192), interpolation=cv2.INTER_AREA)
    norm = (crop.astype(np.float32) / 255 - 0.5) / 0.25  # src/cubecoach/models.py: _Norm 과 같은 계산

    rng = np.random.default_rng(3)
    while True:
        sc, roi = sample_scene(rng)
        if roi[2] < sc.unit * 3 * 3.2:
            break
    f = render(sc, roi, rng, res=192)

    fig, ax = plt.subplots(2, 3, figsize=(11, 7.2))
    items = [
        (boxed, f"실제 원본 {W}x{H} + 관심영역"), (crop, "관심영역만 잘라 192x192"),
        (norm[..., 0], "모델 입력 (x−0.5)/0.25, 빨강 채널"),
        (f.img, "합성 렌더 (입력과 같은 192x192)"),
        (cv2.resize(PAL[f.seg], (192, 192), interpolation=cv2.INTER_NEAREST), "정답 분류 지도 48x48 (다수결)"),
        (f.heat, "스티커 중심 지도 48x48 (가우시안)"),
    ]
    for k, (a, (img, title)) in enumerate(zip(ax.ravel(), items)):
        if k == 2:  # 정규화한 입력: 값 범위 −2~2 를 색 막대와 함께
            im = a.imshow(img, cmap="coolwarm", vmin=-2, vmax=2)
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
        else:
            a.imshow(img, cmap="inferno" if img.ndim == 2 else None)
        a.set_title(title, fontsize=11)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "preprocess.png", dpi=120)


if __name__ == "__main__":
    main()
