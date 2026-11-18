"""실제 촬영 영상에서 '큐브가 없는' 조각(옷, 손, 벽)을 잘라 합성 학습의 배경으로 쓴다.

합성 배경만으로 학습한 모델은 실제 영상에서 회청색 플리스를 파랑으로, 손을 주황·흰색으로 읽었다.
실제 배경을 '그 밖' 분류의 예로 넣어 이 차이를 줄인다. 라벨이 필요 없다(큐브가 없는 조각만 고르므로 전부 '그 밖').

큐브가 들어간 조각은 버린다: 큐브 칸은 채도가 아주 높다(옷·피부는 0.5 미만).
시간 뒷부분(--holdout 초 이후)은 쓰지 않고 실제 영상 평가용으로 남긴다.

실행: .venv/bin/python scripts/extract_real_bg.py data/raw/cube_unsolved.mov
결과: data/real/bg_patches.npy  (N, 224, 224, 3) uint8 RGB
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = 224


def cube_like(rgb: np.ndarray) -> float:
    """큐브 칸처럼 채도가 아주 높은 픽셀의 비율."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # uint8: H 0~179, S·V 0~255
    return float(((hsv[..., 1] > 150) & (hsv[..., 2] > 110)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="+")
    ap.add_argument("--holdout", type=float, default=300.0, help="이 시각 이후는 평가용으로 남긴다 (초)")
    ap.add_argument("--every", type=float, default=0.5)
    ap.add_argument("--per-frame", type=int, default=6)
    ap.add_argument("--max", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out: list[np.ndarray] = []
    tried = 0
    for path in args.video:
        cap = cv2.VideoCapture(path)
        t = 0.0
        while t < args.holdout:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            for _ in range(args.per_frame):
                s = int(rng.uniform(0.12, 0.6) * H)
                y, x = int(rng.integers(0, H - s)), int(rng.integers(0, W - s))
                crop = rgb[y:y + s, x:x + s]
                tried += 1
                if cube_like(crop) > 0.002:
                    continue
                out.append(cv2.resize(crop, (P, P), interpolation=cv2.INTER_AREA))
            t += args.every
    rng.shuffle(out)
    arr = np.stack(out[: args.max])
    dst = ROOT / "data" / "real" / "bg_patches.npy"
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, arr)
    print(f"시도 {tried}개 중 큐브 없는 조각 {len(out)}개, 저장 {len(arr)}개 → {dst}  ({arr.nbytes / 1e6:.0f}MB)")
    sheet = np.vstack([np.hstack(list(arr[i * 10:(i + 1) * 10])) for i in range(6)])
    cv2.imwrite(str(ROOT / "data" / "real" / "private_reports" / "real_bg_patches.jpg"), cv2.cvtColor(cv2.resize(sheet, None, fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
