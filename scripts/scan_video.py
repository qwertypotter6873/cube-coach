"""실제 영상에서 처음 상태 읽기(면 정면 스캔)를 시험한다.

실행: .venv/bin/python scripts/scan_video.py data/raw/sexy_move.mov --end 15
      .venv/bin/python scripts/scan_video.py data/raw/latest.mov --end 45 --truth latest
      .venv/bin/python scripts/scan_video.py data/raw/latest2.mov --end 64 --truth latest2
정답: scripts/track_video.py 의 SEXY_MOVE_STATE·LATEST_STATE·LATEST2_STATE (사람이 면을 읽고 4^6 조합 검사로 정한 상태)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_video import LATEST2_STATE, LATEST_STATE, SEXY_MOVE_STATE, crop  # noqa: E402

from cubecoach.facescan import FaceScanner, follow_roi, read_face  # noqa: E402
from cubecoach.render3d import N_CLASSES  # noqa: E402
from cubecoach.segnet import CubeSegNet, load_segnet, maps_to  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=15.0)
    ap.add_argument("--truth", default=SEXY_MOVE_STATE)
    ap.add_argument("--weights", default=str(ROOT / "models" / "cube_seg_net.pt"))
    ap.add_argument("--res", type=int, default=192, help="모델 입력 한 변")
    ap.add_argument("--map-n", type=int, default=0, dest="map_n", help="지도를 이 크기로 넓이 평균해 줄인 뒤 읽는다 (0 = 그대로)")
    args = ap.parse_args()
    args.truth = {"sexy": SEXY_MOVE_STATE, "latest": LATEST_STATE, "latest2": LATEST2_STATE}.get(args.truth, args.truth)
    m = load_segnet(args.weights)
    m.eval()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    sc = FaceScanner()
    roi = None
    i = 0
    names = "흰빨초노주파"
    t_done = None
    last_try = -1.0
    while True:
        ok, fr = cap.read()
        t = args.start + i / fps
        i += 1
        if not ok or t > args.end:
            break
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        r = roi or (0.0, 0.0, H / W)
        x = crop(rgb, r, res=args.res)
        with torch.no_grad():
            o = m(torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255)[0]
        logp = torch.log_softmax(o[:N_CLASSES], 0).numpy()
        heat = torch.sigmoid(o[N_CLASSES]).numpy()
        if args.map_n:
            logp, heat = maps_to(logp, heat, args.map_n)
        cls = logp.argmax(0)
        face = read_face(cls, heat)
        if face:
            c = sc.add(face)
            if sc.frames[c] in (1, 3):
                print(f"{t:5.2f}s  {names[c]} 면 {sc.frames[c]}번째: {''.join(names[v] for v in face)}")
        roi = follow_roi(cls, r)
        if sc.ready() and t_done is None and t - last_try >= 0.5:  # 앱처럼 0.5초마다 맞춰 본다
            last_try = t
            t0 = time.time()
            s = sc.solve()
            print(f"{t:5.2f}s  여섯 면 모임 → 맞추기 {'성공' if s else '실패, 다시 보여 달라고 할 면: ' + names[sc.weakest()]}"
                  f" ({(time.time() - t0) * 1000:.0f}ms)")
            if s:
                t_done = t
                print("읽은 상태:", s)
                print("정답 상태:", args.truth, "→", "일치" if s == args.truth else "불일치")
                break
    print("면별 프레임 수:", dict(zip(names, sc.frames.tolist())))
    print("면별 같은 격자 최다 횟수:", dict(zip(names, [sc.agree(c) for c in range(6)])))


if __name__ == "__main__":
    main()
