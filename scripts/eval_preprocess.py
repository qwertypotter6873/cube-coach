"""전처리 전·후 비교 (실제 프레임 40장, 칸 723개): 관심영역 자르기의 효과.

전: 화면 가운데의 높이만 한 정사각형을 통째로 192x192로 줄여 넣는다 (큐브가 작게 보임)
전(멀리): 그 두 배 넓이 — 카메라에서 두 배 멀리 떨어져 큐브가 절반 크기로 찍힌 경우를 흉내 낸다
후: 큐브 둘레 관심영역만 잘라 192x192 (추적 중 앱이 하는 방식)
실행: .venv/bin/python scripts/eval_preprocess.py   결과: reports/eval_preprocess.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_video import crop  # noqa: E402

from cubecoach.render3d import N_CLASSES  # noqa: E402
from cubecoach.segnet import CubeSegNet  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    m = CubeSegNet()
    m.load_state_dict(torch.load(ROOT / "models" / "cube_seg_net.pt", map_location="cpu", weights_only=True))
    m.eval()
    frames = json.loads((ROOT / "data" / "real" / "label_frames" / "frames.json").read_text())
    labels = json.loads((ROOT / "data" / "real" / "labels.json").read_text())
    cap = cv2.VideoCapture(str(ROOT / "data" / "raw" / "cube_unsolved.mov"))
    res = {"전(멀리): 2배 넓게": [0, 0, 0], "전: 화면 전체": [0, 0, 0], "후: 관심영역": [0, 0, 0]}  # 맞음, 놓침(그 밖), 전체
    for f in frames:
        pts = labels[f["file"]]["points"]
        cap.set(cv2.CAP_PROP_POS_MSEC, f["t"] * 1000)
        _, fr = cap.read()
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        cx, cy, s = f["roi"]
        for name, roi in [("전(멀리): 2배 넓게", (cx, cy, 2 * H / W)), ("전: 화면 전체", (0.0, 0.0, H / W)), ("후: 관심영역", tuple(f["roi"]))]:
            x = crop(rgb, roi)
            with torch.no_grad():
                o = m(torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255)[0]
            cls = o[:N_CLASSES].argmax(0).numpy()
            for p in pts:
                # 관심영역 안 좌표 → 화면 좌표(너비 기준, 중심 0) → 이 입력 안 좌표
                u, v = cx + (p["x"] - 0.5) * s, cy + (p["y"] - 0.5) * s
                i = int(np.clip(((v - roi[1]) / roi[2] + 0.5) * 48, 0, 47))
                j = int(np.clip(((u - roi[0]) / roi[2] + 0.5) * 48, 0, 47))
                res[name][0] += int(cls[i, j] == p["c"])
                res[name][1] += int(cls[i, j] == 7)
                res[name][2] += 1
    out = {k: {"sticker_acc": a / n, "missed": miss / n, "n": n} for k, (a, miss, n) in res.items()}
    (ROOT / "reports" / "eval_preprocess.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k, v in out.items():
        print(f"{k:<12} 칸 색 정확도 {v['sticker_acc']:.3f}  놓침 {v['missed']:.3f}  (칸 {v['n']})")


if __name__ == "__main__":
    main()
