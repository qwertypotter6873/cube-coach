"""분할 모델용 합성 데이터를 만들어 디스크에 저장한다 (메모리에 한꺼번에 올리지 않는다).

실행: .venv/bin/python scripts/make_seg_data.py              v2 (기본): data/synth3d_v2, 20,000장
      RENDER=v1 .venv/bin/python scripts/make_seg_data.py    v1: data/synth3d, 16,000장 (src/cubecoach/render3d_v1.py)
      RENDER=v3 .venv/bin/python scripts/make_seg_data.py    v3: data/synth3d_v3, 20,000장 (v2 + 층을 돌릴 때 그 층만 번짐)
결과: data/<폴더>/{train,val,test}_{img,seg,heat,centers,vis,colors}.npy
렌더러 버전(RENDER)이 폴더·장수의 기본값을 정한다. SYNTH_DIR·N_TRAIN 으로 바꿀 수는 있지만, 렌더러는 RENDER 로만 바뀐다.
v2 는 실제 촬영 배경 조각(data/real/bg_patches.npy, scripts/extract_real_bg.py)이 있어야 한다. 이 조각은 촬영자가 나오는
영상에서 잘라서 공개하지 않는다. 없으면 합성 배경만 쓰는 다른 실험이 되므로 멈춘다 (ALLOW_NO_REAL_BG=1 이면 그대로 진행).
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import os

import numpy as np

# v1: 검은 몸체 스티커 큐브 + 합성 배경 (data/synth3d, 16000장)
# v2: + 스티커리스 큐브, 실제 촬영 배경·피부 조각, 실제 큐브 색, 움직임 번짐 (data/synth3d_v2, 20000장)
VERSION = os.environ.get("RENDER", "v2")
if VERSION == "v1":
    from cubecoach.render3d_v1 import STRIDE, render, sample_scene
elif VERSION in ("v2", "v3"):
    from cubecoach.render3d import STRIDE, real_bg, render, sample_scene
else:
    raise SystemExit(f"RENDER 는 v1 또는 v2 여야 합니다: {VERSION}")
DEFAULTS = {"v1": ("synth3d", 16000), "v2": ("synth3d_v2", 20000), "v3": ("synth3d_v3", 20000)}
RENDER_KW = {"version": 3} if VERSION == "v3" else {}  # v1·v2 는 예전 호출 그대로
OUT = Path(__file__).resolve().parents[1] / "data" / os.environ.get("SYNTH_DIR", DEFAULTS[VERSION][0])
RES = 192
N_TRAIN = int(os.environ.get("N_TRAIN", str(DEFAULTS[VERSION][1])))
SPLITS = {"train": (N_TRAIN, 10_000), "val": (1000, 20_000), "test": (1500, 30_000)}  # 이름: (장수, 시드)
WORKERS = 8


def _job(args: tuple[str, int, int, int]) -> int:
    split, start, stop, seed = args
    arrs = {k: np.load(OUT / f"{split}_{k}.npy", mmap_mode="r+") for k in ("img", "seg", "heat", "centers", "vis", "colors")}
    rng = np.random.default_rng(seed)
    for i in range(start, stop):
        scene, roi = sample_scene(rng)
        fr = render(scene, roi, rng, res=RES, **RENDER_KW)
        arrs["img"][i], arrs["seg"][i] = fr.img, fr.seg
        arrs["heat"][i] = (fr.heat * 255).astype(np.uint8)
        arrs["centers"][i], arrs["vis"][i], arrs["colors"][i] = fr.centers, fr.vis, scene.colors
    for a in arrs.values():
        a.flush()
    return stop - start


def main() -> None:
    if VERSION in ("v2", "v3") and real_bg()[0] is None and os.environ.get("ALLOW_NO_REAL_BG") != "1":
        raise SystemExit("data/real/bg_patches.npy 가 없습니다. v2 는 실제 촬영 배경 조각이 있어야 보고한 실험과 같아집니다"
                         " (scripts/extract_real_bg.py, 촬영 영상 필요). 합성 배경만으로 만들려면 ALLOW_NO_REAL_BG=1.")
    print(f"렌더러 {VERSION} → {OUT} (학습 {N_TRAIN}장)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    n_lab = RES // STRIDE
    for split, (n, seed) in SPLITS.items():
        if len(sys.argv) > 1 and split not in sys.argv[1:]:
            continue
        shapes = {
            "img": ((n, RES, RES, 3), np.uint8), "seg": ((n, n_lab, n_lab), np.uint8),
            "heat": ((n, n_lab, n_lab), np.uint8), "centers": ((n, 54, 2), np.float32),
            "vis": ((n, 54), np.float32), "colors": ((n, 54), np.uint8),
        }
        for k, (shape, dtype) in shapes.items():
            np.lib.format.open_memmap(OUT / f"{split}_{k}.npy", mode="w+", dtype=dtype, shape=shape).flush()
        step = -(-n // (WORKERS * 4))
        jobs = [(split, s, min(s + step, n), seed + j) for j, s in enumerate(range(0, n, step))]
        t0 = time.time()
        with ProcessPoolExecutor(WORKERS) as ex:
            done = sum(ex.map(_job, jobs))
        print(f"{split}: {done}장, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
