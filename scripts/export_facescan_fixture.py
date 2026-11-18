"""TS 면 읽기(web/src/vision/facescan.ts)가 파이썬(src/cubecoach/facescan.py)과 같은지 시험할 정답값을 만든다.

출력: web/test/fixtures/facescan.json
  cases  : 공개 샘플과 같은 촬영(sexy_move.mov)의 프레임에서 뽑은 모델 지도(48x48 분류 확률·중심 지도, 픽셀은 없음)
           + 파이썬 read_face 결과
           - 7장: 처음 만든 정답값과 같은 프레임·관심영역
           - 1장: 7.7초 지도의 오른쪽 아래(큐브와 떨어진 곳)에 뒤쪽 책장처럼 스티커로 읽히는 덩어리(흰·초록 4x18칸,
             중심 봉우리 6개)를 붙인 것. 고치기 전 규칙은 이 지도에서 면을 못 읽고(None), 지금 규칙은 붙이기 전과 같은 면을 읽는다.
           공개 동의 범위 밖인 latest.mov·latest2.mov 의 지도는 넣지 않는다.
  seen, truth : sexy_move.mov 여섯 면과 정답 상태 (맞추기 시험용)
  replay : latest2.mov 0~33초를 앱과 같은 흐름(관심영역 따라가기, 0.5초마다 맞추기)으로 돌린 파이썬 읽기 순서
           (읽은 9칸 색과 시각뿐), 맞추기를 시도한 시각과 성공 여부, 성공한 상태(큐브 색 54칸).
           TS 시험이 같은 순서로 넣어 시도마다 같은 결과가 나오는지 본다. 이 영상이 있어야 다시 만들 수 있다.

실행: .venv/bin/python scripts/export_facescan_fixture.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from track_video import LATEST2_STATE, SEXY_MOVE_STATE, crop  # noqa: E402

from cubecoach.cube import COLORS  # noqa: E402
from cubecoach.facescan import FaceScanner, follow_roi, read_face  # noqa: E402
from cubecoach.render3d import N_CLASSES  # noqa: E402
from cubecoach.segnet import CubeSegNet  # noqa: E402

FRAMES = {0.3: (0.0, 0.05, 0.5), 1.6: (0.0, 0.0, 0.5), 3.2: (0.0, 0.0, 0.5), 4.6: (0.02, 0.0, 0.5),
          5.6: (0.02, 0.0, 0.5), 7.7: (0.02, 0.0, 0.5), 12.5: (0.0, 0.0, 0.66)}  # sexy_move.mov 시각: 관심영역
BLOB = (42, 28, 4, 18)  # 붙이는 덩어리: 왼쪽 위 x, y, 너비, 높이 (지도 칸)


def with_blob(logp: np.ndarray, heat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """큐브와 떨어진 곳에 스티커로 읽히는 세로 띠(흰·초록 번갈아)를 붙인다. 왼쪽 줄에 3칸마다 중심 봉우리."""
    L, H = logp.copy(), heat.copy()
    x0, y0, w, h = BLOB
    for dx in range(w):
        p = np.full(8, 0.03 / 7, dtype=np.float32)
        p[(0, 2)[dx % 2]] = 0.97
        for dy in range(h):
            L[:, y0 + dy, x0 + dx] = np.log(p)
            H[y0 + dy, x0 + dx] = 0.9 if dx == 0 and dy % 3 == 1 else 0.2
    return L, H


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def maps(m: CubeSegNet, rgb: np.ndarray, roi) -> tuple[np.ndarray, np.ndarray]:
    x = crop(rgb, roi)
    with torch.no_grad():
        o = m(torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255)[0]
    return torch.log_softmax(o[:N_CLASSES], 0).numpy().astype(np.float32), torch.sigmoid(o[N_CLASSES]).numpy().astype(np.float32)


def main() -> None:
    m = CubeSegNet()
    m.load_state_dict(torch.load(ROOT / "models" / "cube_seg_net.pt", map_location="cpu", weights_only=True))
    m.eval()
    cases = []
    cap = cv2.VideoCapture(str(ROOT / "data/raw/sexy_move.mov"))
    for t, roi in FRAMES.items():
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        _, fr = cap.read()
        logp, heat = maps(m, cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), roi)
        f = read_face(logp.argmax(0), heat)
        cases.append({"video": "sexy_move", "t": t, "logp": b64(logp), "heat": b64(heat), "face": f})
        print("sexy_move", t, f)
        if t == 7.7:
            L, H = with_blob(logp, heat)
            fb = read_face(L.argmax(0), H)
            assert fb == f, (fb, f)  # 떨어진 덩어리는 읽기를 바꾸지 않는다
            cases.append({"video": "sexy_move", "t": t, "blob": list(BLOB), "logp": b64(L), "heat": b64(H), "face": fb})
            print("sexy_move", t, "+ 덩어리", fb)

    # 앱과 같은 흐름으로 latest2 처음 33초
    cap = cv2.VideoCapture(str(ROOT / "data/raw/latest2.mov"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sc, roi, i, reads, attempts, last_try, solved = FaceScanner(), None, 0, [], [], -1.0, None
    while solved is None:
        ok, fr = cap.read()
        t = i / fps
        i += 1
        if not ok or t > 33:
            break
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        r = roi or (0.0, 0.0, rgb.shape[0] / rgb.shape[1])
        logp, heat = maps(m, rgb, r)
        cls = logp.argmax(0)
        f = read_face(cls, heat)
        if f:
            sc.add(f)
            reads.append([round(t, 4), f])
        roi = follow_roi(cls, r)
        if sc.ready() and t - last_try >= 0.5:
            last_try = t
            s = sc.solve()
            attempts.append([round(t, 4), bool(s)])
            if s:
                solved = {"t": round(t, 4), "state": [COLORS.index(c) for c in s]}
    assert solved and solved["state"] == [COLORS.index(c) for c in LATEST2_STATE], solved
    print("latest2 맞추기", solved["t"], "초, 읽은 프레임", len(reads))

    seen = {"R": "RWBBRRWWY", "B": "BGBYBGBGY", "O": "OORGORWRG", "G": "GBOBGOWOW", "W": "YYGBWROWR", "Y": "YYOOYYRWG"}
    out = ROOT / "web" / "test" / "fixtures" / "facescan.json"
    json.dump({"cases": cases, "seen": {k: [COLORS.index(c) for c in v] for k, v in seen.items()},
               "truth": [COLORS.index(c) for c in SEXY_MOVE_STATE],
               "replay": {"video": "latest2", "reads": reads, "attempts": attempts, "solved": solved}}, open(out, "w"))
    print(out, f"{out.stat().st_size / 1024:.0f}KB")


if __name__ == "__main__":
    main()
