"""README용 학습곡선(v1·v2, 베이스라인)과 합성 시험셋 혼동행렬(v2).

실행: .venv/bin/python scripts/fig_training.py   결과: reports/learning_curves.png, reports/confusion_synth.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_seg as T  # noqa: E402

from cubecoach.render3d import N_CLASSES  # noqa: E402
from cubecoach.segnet import CubeSegNet  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
NAMES = ["흰", "빨", "초", "노", "주", "파", "몸체", "그 밖"]


def main() -> None:
    v1 = json.loads((ROOT / "reports" / "train_seg_v1.json").read_text())
    v2 = json.loads((ROOT / "reports" / "train_seg.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for name, res, style in [("CubeSegNet v1 (합성만)", v1["CubeSegNet (제안)"], "--"), ("CubeSegNet v2", v2["CubeSegNet (제안)"], "-"),
                             ("PixelMLP v2 (베이스라인)", v2["PixelMLP (칸 평균색만, 베이스라인)"], ":")]:
        log = res["log"]
        ep = [r["epoch"] for r in log]
        ax[0].plot(ep, [r["ce"] for r in log], style, label=name)
        ax[1].plot(ep, [r["val_miou"] for r in log], style, label=name)
        ax[2].plot(ep, [r["val_sticker_acc"] for r in log], style, label=name)
    for a, t in zip(ax, ["학습 손실 (분류 교차엔트로피)", "검증 mIoU", "검증 칸 색 정확도"]):
        a.set_title(t)
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    ax[2].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "learning_curves.png", dpi=130)

    m = CubeSegNet()
    m.load_state_dict(torch.load(ROOT / "models" / "cube_seg_net.pt", map_location="cpu", weights_only=True))
    m.to(T.DEVICE)
    te = T.load("test")
    cls, _ = T.infer(m, te["img"])
    seg = np.asarray(te["seg"])
    cm = np.bincount(seg.ravel().astype(int) * N_CLASSES + cls.ravel(), minlength=N_CLASSES**2).reshape(N_CLASSES, N_CLASSES)
    rate = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, a = plt.subplots(figsize=(6.2, 5.2))
    a.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if rate[i, j] >= 0.005:
                a.text(j, i, f"{rate[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if rate[i, j] > 0.5 else "black")
    a.set_xticks(range(N_CLASSES), NAMES)
    a.set_yticks(range(N_CLASSES), NAMES)
    a.set_xlabel("예측")
    a.set_ylabel("정답")
    a.set_title("합성 시험셋 칸 단위 혼동행렬 (CubeSegNet v2, 행 정규화)")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "confusion_synth.png", dpi=130)


if __name__ == "__main__":
    main()
