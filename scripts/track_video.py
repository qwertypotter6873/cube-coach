"""실제 촬영 영상에서 연속 추적을 돌려 본다 (인식 모델 + 추적기, 앱과 같은 흐름).

정답이 있는 시험 영상(sexy_move.mov): 처음 8초에 여섯 면을 보여 주고(→ 처음 상태),
R U R' U' 6번, 큐브 고쳐 잡기, F U F' U' 6번을 했다. 두 동작 모두 6번 하면 원래 상태로 돌아온다.
처음 상태는 여섯 면 사진을 읽고, 면 방향 4^6 조합 가운데 유효한 상태가 하나뿐인 것으로 정했다
(scripts/track_video.py 의 SEXY_MOVE_STATE, 확인 방법은 README 5절).

평가
- 판정한 회전 수 (정답 48)
- 회전열 정확도: 1 - (정답 회전열과의 편집 거리 / 48). 정답 회전열(가운데 색 기준)은 '어느 색 면을 오른쪽·위·앞으로
  잡았는가'에 따라 달라진다. 두 동작 사이에 고쳐 잡으므로 앞·뒤 동작 각각 24가지 잡는 방향(24 x 24 = 576가지)
  가운데 판정 결과와 가장 가까운 것을 정답으로 삼는다. 무작위 판정으로도 이 방식의 정확도는 0.17 정도 나온다(v1 결과).
- 마지막 상태가 처음 상태와 같은가

실행: .venv/bin/python scripts/track_video.py data/raw/sexy_move.mov --ts --video
결과: reports/track_real_<태그>.json (+ --video 면 reports/track_real_<태그>.mp4)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_tracking import TsTracker, edit_distance, merge  # noqa: E402

from cubecoach.cube import COLORS, apply  # noqa: E402
from cubecoach.render3d import N_CLASSES  # noqa: E402
from cubecoach.segnet import CubeSegNet, load_segnet, maps_to  # noqa: E402
from cubecoach.tracker import Tracker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEXY_MOVE_STATE = "GRRYWWYBOBRYWRWRBWOOWBGOGBWOYGYYWYORRRGOOROGWBGYGBGBYB"
# 정답 회전열 (가운데 색, 시계 방향 90도 단위). 색: 0 흰 1 빨 2 초 3 노 4 주 5 파
# 정한 방법: 촬영자가 F U F' U' 횟수를 기억하지 못해, 89.2초 프레임에서 사람이 읽은 두 면(16칸)과
# '처음 상태 + 섹시무브 m수 + F U F' U' n수' (m 0~32, n 0~40) 전부를 견줬다. 16칸이 모두 맞는 것은
# m ≡ 1 (mod 24), n ≡ 4 (mod 24) 뿐이었다. 섹시무브 구간은 6.6초(1초에 약 4수)라 25수,
# F U F' U' 구간은 60초 동안 계속 돌리므로 28수(7번)로 정했다 (4수로는 60초를 설명할 수 없다).
# 덧붙은 흰 한 번(슬롯 U)은 13초쯤 큐브를 떨어뜨렸다 다시 들 때 층이 돌아간 것이다 (촬영자 확인, 2026-11-17).
# 그래서 섹시무브 구간(A)은 '처음 상태 + U'에서 시작한다. 이 한 수의 색·방향은 89.2초 대조(mod 24)와 추적기 판정에서 왔다.
DROP_TURN = [(0, 1)]  # 흰 시계 방향 90도, 12.6~15초 사이 (떨어뜨림·다시 들기)
SEXY_MOVE_GT = {
    # 이름: (시작 초, 끝 초, 처음 상태에 앞서 적용된 슬롯 회전, 정답 회전열)
    "A": (19.5, 27.5, "U", [(0, 1), (1, 1), (0, 3), (1, 3)] * 6),  # (흰 빨 흰' 빨') x6
    "B": (30.5, 90.0, "U", [(4, 1), (2, 1), (4, 3), (2, 3)] * 7),  # (주 초 주' 초') x7
}
SEXY_MOVE_GT["AB"] = (19.5, 90.0, "U", SEXY_MOVE_GT["A"][3] + SEXY_MOVE_GT["B"][3])
SEXY_MOVE_GT["all"] = (0.0, 90.0, "", DROP_TURN + SEXY_MOVE_GT["AB"][3])
# 층을 일부러 돌리지 않은 구간: 정답 회전은 떨어뜨릴 때 돌아간 한 수뿐(Z1·Z1b) 또는 없음(Z2). 나머지 판정은 가짜다.
#  Z1: 여섯 면 보여 주기와 들고 있기·떨어뜨림, Z2: 섹시무브를 마친 뒤 고쳐 잡고 첫 F 전까지
SEXY_MOVE_GT["Z1"] = (0.0, 19.5, "", DROP_TURN)
SEXY_MOVE_GT["Z2"] = (26.6, 34.5, "U", [])
SEXY_MOVE_GT["Z1b"] = (8.2, 19.5, "", DROP_TURN)  # Z1 가운데 앱에서 실제로 추적하는 부분 (면 읽기가 끝난 뒤)
# 앱과 같은 흐름: 면 읽기가 끝나는 7.7초(scripts/scan_video.py)부터 끝까지 이어서. 2026-11-18 자세 이탈 게이트를 견줄 때 더했다.
SEXY_MOVE_GT["app"] = (7.7, 90.0, "", DROP_TURN + SEXY_MOVE_GT["AB"][3])
SEXY_MOVE_FINAL = "U L F L' F'"  # 전체를 마친 상태 = 처음 상태에 이 슬롯 회전을 한 것

# 두 번째 시험 영상(latest.mov, 194초, 노트북 웹캠, 2026-11-17 촬영). 문턱을 고른 뒤에 받은 영상이라 시험에만 쓴다.
# 처음 상태: 0~18초에 보여 준 다섯 면(노랑은 보여 주지 않음)을 사람이 읽고, 면 방향 4^5 x 거울 여부 가운데 유효한
#   조합이 하나뿐임을 확인했다. 노랑 면 후보 3개 가운데 22.9초·107.8초 프레임과 맞는 것으로 정했다.
# 동작: 촬영 안내대로 R U R' U' x6(초록 위, 노랑 오른쪽 = 슬롯 D F D' F'), L' U' L U x6(흰 위, 초록 앞).
#   두 동작 모두 쉬는 순간마다 카메라 쪽 면이 '처음 상태 + k번' 예측과 맞는 것을 k = 1~6 모두 확인했다.
#   52.5초·78초·80초·100초·104.4초·105.5초·106.4초·107.8초 면이 처음 상태와 같다(층을 돌리지 않은 구간 확인).
# 108초 뒤는 촬영자가 마음대로 돌렸고 끝에 면을 보여 주지 않아 정답이 없다(free: 판정 수·추적 비율만 본다).
# 경계는 추적기를 돌리기 전에 프레임만 보고 정했다.
LATEST_STATE = "RGGYWORYWRBYGRRWBOWOGRGOWBOOWGYYRYOYBGBWOWOBBRRYYBGGWB"
LATEST_R = [(3, 1), (2, 1), (3, 3), (2, 3)] * 6  # (노 초 노' 초') x6, 첫 회전 52.7초, 끝 71초
LATEST_L = [(4, 3), (0, 3), (4, 1), (0, 1)] * 6  # (주' 흰' 주 흰) x6, 첫 회전 85.2초, 끝 95.5초
LATEST_GT = {
    # 이름: (시작 초, 끝 초, 정답 회전열 또는 None = 정답 없음). 모든 구간이 처음 상태에서 시작해 처음 상태로 끝난다.
    "Z1": (0.0, 52.5, []),      # 면 보여 주기 + 통째 돌리기 (층 회전 없음)
    "Z1b": (18.3, 52.5, []),    # 면 보여 주기를 마친 뒤 (앱이라면 추적을 시작하는 때)
    "R": (52.0, 72.8, LATEST_R),  # 잡은 자세가 안정된 52초부터 (첫 4수를 0.6초 안에 빠르게 돌렸다)
    "ZR": (18.3, 72.8, LATEST_R),  # 통째 돌리기부터 이어서
    "Z2": (72.8, 85.0, []),     # 통째 돌리기 + 면 보여 주기
    "L": (83.0, 96.8, LATEST_L),  # 잡은 자세가 안정된 83초부터
    "Z3": (96.8, 108.0, []),    # 면 다시 보여 주기 (노랑 포함 다섯 면)
    "all": (0.0, 108.0, LATEST_R + LATEST_L),
    "free": (108.0, None, None),
}

# 세 번째 시험 영상(latest2.mov, 190초 가운데 촬영자 요청대로 180초까지만, 2026-11-17 촬영). 천천히 돌리기만 시험한다.
# 처음 상태: 0~51초에 여섯 면을 차례로 보여 주었다(흰 노 빨 주 초 파). 사람이 읽은 여섯 면의 방향 4^6 x 거울 여부 가운데
#   유효한 조합이 하나뿐이었다(거울상 0개).
# 동작: 흰 위·초록 앞으로 잡고 R U R' U' x6(64.5~97초), 잠시 멈춘 뒤 L' U' L U x6(106.5~131.5초), 한 수마다 멈춤.
#   카메라 쪽(파랑) 면이 매 수 또는 매 회차 뒤 '처음 상태 + 그때까지의 회전' 예측과 맞는 것을 확인했다.
#   64.5초, 97~106초, 132~135초 면과 136~147초에 다시 보여 준 여섯 면(흰 노 빨 주 초)이 처음 상태와 같다.
# 148초 뒤는 다시 자유롭게 돌렸고 면을 보여 주지 않아 정답이 없다(free, 180초에서 자른다).
# 경계는 추적기를 돌리기 전에 프레임만 보고 정했다.
LATEST2_STATE = "YOOOWGRROBWGYRBGYBWGWWGBYYYOGROYRWBRBBBOORGWGWYOWBGYRR"
LATEST2_R = [(1, 1), (0, 1), (1, 3), (0, 3)] * 6  # (빨 흰 빨' 흰') x6
LATEST2_L = [(4, 3), (0, 3), (4, 1), (0, 1)] * 6  # (주' 흰' 주 흰) x6
LATEST2_GT = {
    "Z1": (0.0, 64.3, []),        # 여섯 면 보여 주기 + 잡기 (층 회전 없음)
    "Z1b": (51.0, 64.3, []),      # 면 보여 주기를 마친 뒤
    "R": (64.0, 97.5, LATEST2_R),
    "Z2": (97.5, 106.3, []),      # 두 동작 사이에 들고 있기
    "L": (104.0, 134.5, LATEST2_L),
    "Z3": (132.5, 148.0, []),     # 면 다시 보여 주기
    "RL": (64.0, 134.5, LATEST2_R + LATEST2_L),  # 두 동작을 이어서
    "all": (0.0, 148.0, LATEST2_R + LATEST2_L),
    "free": (148.0, 180.0, None),
    # 결과를 본 뒤 더한 구간: 앱처럼 면 보여 주기를 마친 뒤(51초)부터 두 동작 끝까지 이어서. 지금 기본으로만 쟀다.
    "app": (51.0, 134.5, LATEST2_R + LATEST2_L),
    # 면 읽기를 고친 뒤(2026-11-18) 앱이 읽기를 마치는 시각(32.86초, scripts/scan_video.py --truth latest2)부터 이어서.
    # 51초까지는 면을 마저 보여 주고 고쳐 잡는 구간이다. 이것도 결과를 본 뒤 더했고, 지금 기본으로만 쟀다.
    "afterscan": (32.86, 134.5, LATEST2_R + LATEST2_L),
}
VIDEO_GT = {"latest": (LATEST_STATE, LATEST_GT), "latest2": (LATEST2_STATE, LATEST2_GT)}
HEX = [(240, 240, 236), (214, 35, 58), (4, 168, 70), (233, 224, 44), (250, 106, 44), (4, 100, 188)]


def lcs(a: list, b: list) -> int:
    """두 회전열의 최장 공통 부분열 길이 (가짜 회전 셀 때 정답 회전과 짝짓기)"""
    t = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            t[i + 1][j + 1] = t[i][j] + 1 if x == y else max(t[i][j + 1], t[i + 1][j])
    return t[-1][-1]


def crop(frame: np.ndarray, roi, res: int = 192) -> np.ndarray:
    """roi(프레임 너비 기준, 중심 0)를 잘라 res x res. 화면 밖은 검게."""
    H, W = frame.shape[:2]
    side = roi[2] * W
    sx, sy = W / 2 + roi[0] * W - side / 2, H / 2 + roi[1] * W - side / 2
    k = res / side
    M = np.array([[k, 0, -sx * k], [0, k, -sy * k]], dtype=np.float32)
    return cv2.warpAffine(frame, M, (res, res), flags=cv2.INTER_AREA, borderValue=(0, 0, 0))


def expected_sequences(algs: tuple[str, ...] = ("R U R' U'", "F U F' U'")) -> list[list[tuple[int, int]]]:
    """잡는 방향 조합마다 'R U R' U' x6, F U F' U' x6'의 가운데 색 회전열."""
    from itertools import permutations

    # 잡는 방향 = (위 색, 앞 색). 오른쪽 색은 위 x 앞.
    solved_pos = {"U": (0, 1, 0), "D": (0, -1, 0), "R": (1, 0, 0), "L": (-1, 0, 0), "F": (0, 0, 1), "B": (0, 0, -1)}
    color_of = {"U": 0, "R": 1, "F": 2, "D": 3, "L": 4, "B": 5}  # 표준 배색 (처음 상태도 표준 배색)
    vec2face = {v: f for f, v in solved_pos.items()}
    out = []
    for up, front in permutations("URFDLB", 2):
        u, f = np.array(solved_pos[up]), np.array(solved_pos[front])
        if abs(u @ f) > 0:
            continue
        r = np.cross(u, f)
        right = vec2face[(int(r[0]), int(r[1]), int(r[2]))]
        m = {"U": color_of[up], "F": color_of[front], "R": color_of[right]}
        out.append(m)
    seq = lambda m, alg: [(m[t[0]], 3 if t.endswith("'") else 1) for t in (alg + " ") .split() * 6]
    if len(algs) == 1:
        return [seq(a, algs[0]) for a in out]
    # 두 동작 사이에 큐브를 고쳐 잡으므로, 앞 동작과 뒤 동작의 잡는 방향을 따로 고른다 (24 x 24)
    return [seq(a, algs[0]) + seq(b, algs[1]) for a in out for b in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--state", default=SEXY_MOVE_STATE)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1, help="프레임 건너뛰기 (2 = 15fps)")
    ap.add_argument("--weights", default=str(ROOT / "models" / "cube_seg_net.pt"))
    ap.add_argument("--ts", action="store_true", help="브라우저용 TS 추적기 사용")
    ap.add_argument("--f", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0, help="추적기 탐색 난수 시드")
    ap.add_argument("--cover", action="store_true", help="(TS) 덮는 비율로 자세 다시 찾기 켜기 (기본 끔, 효과 없었음)")
    ap.add_argument("--no-cover", action="store_true", help="(옛 옵션, 기본이 끔)")
    ap.add_argument("--settle", type=int, default=0, help="(TS) 자세를 새로 잡은 뒤 회전 판정을 쉬는 안정 프레임 수 (기본 0 = 끔)")
    ap.add_argument("--fast", action="store_true", help="(TS) 큐브를 옮길 때 빨리 따라가기 켜기 (세부 스위치 전부)")
    ap.add_argument("--follow", default="", help="(TS) 빨리 따라가기 세부 스위치만 켜기, 쉼표로 (predict,jump,relock,fastLost,wide,calmAfterLock)")
    ap.add_argument("--traj", action="store_true", help="프레임마다 자세 중심·일치율·진단값을 결과에 남긴다")
    ap.add_argument("--res", type=int, default=192,
                    help="모델 입력 한 변 (기본 192 = 지금 앱; 지도는 res/4. 입력 256 실험 갈래는 --res 256 --map-n 48, README 5.2절)")
    ap.add_argument("--map-n", type=int, default=0, dest="map_n",
                    help="지도를 이 크기로 넓이 평균해 줄여 추적기에 넘긴다 (0 = 그대로, 192 입력이면 이미 48x48)")
    ap.add_argument("--opts", default="", help='(TS) 추적기 설정 JSON, 예 {"turnVsBody": 8}')
    ap.add_argument("--bank", type=int, default=1500, help="(TS) 전체 탐색 회전 뱅크 크기 (앱과 같은 1500)")
    ap.add_argument("--pan", type=float, default=0.0,
                    help="큐브를 화면에서 빠르게 옮기는 시늉: 프레임마다 보는 창을 사인파로 움직인다 (화면 너비의 몇 배, 예 0.25)")
    ap.add_argument("--pan-hz", type=float, default=0.25, help="--pan 의 왕복 주기 (Hz)")
    ap.add_argument("--jump", type=float, default=0.0, help="2초마다 보는 창을 화면 너비의 이만큼 순간이동 (예 0.3)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--alg", default=None, help="(정답이 없는 영상) 이 구간에서 한 동작 하나 (6번 반복)")
    ap.add_argument("--segment", choices=list(SEXY_MOVE_GT) + [f"{v}_{k}" for v, (_, g) in VIDEO_GT.items() for k in g], default=None,
                    help="sexy_move.mov 의 정답 구간, latest_* 는 latest.mov, latest2_* 는 latest2.mov 의 정답 구간")
    ap.add_argument("--video-out", action="store_true", dest="video_out")
    args = ap.parse_args()

    gt = None
    vid = args.segment.split("_", 1)[0] if args.segment and args.segment.split("_", 1)[0] in VIDEO_GT else None
    latest = vid is not None
    if latest:
        vstate, vgt = VIDEO_GT[vid]
        args.start, args.end, gt = vgt[args.segment.split("_", 1)[1]]
        args.state = vstate
    elif args.segment:
        args.start, args.end, pre, gt = SEXY_MOVE_GT[args.segment]
        if pre:
            args.state = apply(args.state, pre)
    torch.set_num_threads(4)
    model = load_segnet(args.weights)  # 폭은 가중치에서 알아낸다
    model.eval()
    colors = np.array([COLORS.index(c) for c in args.state])
    cap = cv2.VideoCapture(args.video)
    aspect = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    tracker = (
        TsTracker(colors, f=args.f, seed=args.seed, cover=args.cover and not args.no_cover,
                  settle=args.settle, fast=args.fast, bank=args.bank,
                  follow=[k for k in args.follow.split(",") if k], aspect=aspect,
                  opts=json.loads(args.opts) if args.opts else None)
        if args.ts else Tracker(colors, f=args.f, seed=args.seed)
    )

    def shift(t: float) -> tuple[float, float]:
        """큐브가 화면에서 빠르게 움직이는 시늉: 시각 t에 보는 창을 얼마나 옮길지 (화면 너비 기준)."""
        dx = dy = 0.0
        if args.pan:
            w = 2 * np.pi * args.pan_hz
            dx += args.pan * np.sin(w * t)
            dy += 0.6 * args.pan * np.sin(2 * w * t)  # 8자로 움직인다
        if args.jump:
            step = int(t // 2)
            dx += args.jump * (1 if step % 2 else -1)
            dy += args.jump * 0.5 * (1 if step % 4 >= 2 else -1)
        return float(dx), float(dy)

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    tag = args.tag or Path(args.video).stem + ("_ts" if args.ts else "_py")
    writer = None
    if args.video_out:
        writer = cv2.VideoWriter(str(ROOT / "data" / "real" / "private_reports" / f"track_real_{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (960, 640))
    i = 0
    times = []
    quality = []
    traj = []
    moving_frames = 0
    t0 = time.time()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = args.start + i / fps
        i += 1
        if args.end and t > args.end:
            break
        if (i - 1) % args.stride:
            continue
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        # 놓쳤을 때 찾을 영역은 추적기가 정한다 (앱과 같다). 파이썬 추적기는 가운데 정사각형.
        lost = getattr(tracker, "lost_roi", None) or (0.0, 0.0, H / W)
        roi = tracker.roi() if tracker.R is not None else lost
        # 앱과 같게: 관심영역이 화면 밖으로 거의 다 나가지 않게 당긴다 (web/src/vision/camera.ts: clampRoi)
        side = min(roi[2], 1.6)
        mx, my = 0.5 + side / 2 - min(side, 1.0) / 3, H / W / 2 + side / 2 - min(side, H / W) / 3
        roi = (min(max(roi[0], -mx), mx), min(max(roi[1], -my), my), side)
        dx, dy = shift(t)
        x = crop(rgb, (roi[0] - dx, roi[1] - dy, roi[2]), res=args.res)
        with torch.no_grad():
            o = model(torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255)[0]
        logp = torch.log_softmax(o[:N_CLASSES], 0).numpy()
        heat = torch.sigmoid(o[N_CLASSES]).numpy()
        if args.map_n:
            logp, heat = maps_to(logp, heat, args.map_n)
        tracker.update(logp, heat, roi)
        times.append(t)
        quality.append(tracker.quality if tracker.R is not None else 0.0)
        moving_frames += int(bool(getattr(tracker, "moving", False)))
        if args.traj:
            T = getattr(tracker, "T", None)
            dg = getattr(tracker, "diag", {}) or {}
            uv = [round(float(args.f * T[0] / T[2]), 4), round(float(args.f * T[1] / T[2]), 4), round(float(T[2]), 2)] if T is not None else None
            traj.append([round(t, 3), round(float(tracker.quality), 3) if tracker.R is not None else None, uv,
                         round(dx, 4), round(dy, 4), round(dg.get("blobOff", -1), 3), round(dg.get("speed", 0), 3),
                         round(dg.get("jumpGain", 0), 2), int(tracker.layer), round(dg.get("bodyDeg", 0), 2)])
        if writer is not None:
            vis = cv2.resize(fr, (960, 640))
            s = 960 / W
            side, cx, cy = roi[2] * W * s, (W / 2 + (roi[0] - dx) * W) * s, (H / 2 + (roi[1] - dy) * W) * s
            cv2.rectangle(vis, (int(cx - side / 2), int(cy - side / 2)), (int(cx + side / 2), int(cy + side / 2)), (255, 255, 255), 1)
            txt = f"t={t:5.1f}s q={tracker.quality:.2f} moves={len(tracker.events)} layer={tracker.layer}"
            cv2.putText(vis, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(vis)
    if writer is not None:
        writer.release()
    if hasattr(tracker, "close"):
        tracker.close()
    elapsed = time.time() - t0

    det = [(int(e.center), int(e.quarter_turns)) for e in tracker.events]
    d = merge(det)
    no_gt = latest and gt is None  # latest_free: 정답 없음
    if gt is not None:
        exps = [merge(gt)]
    elif no_gt:
        exps = [[]]
    else:
        exps = [merge(s) for s in (expected_sequences((args.alg,)) if args.alg else expected_sequences())]
    dists = [edit_distance(e, d) for e in exps]
    best = int(np.argmin(dists))
    final = "".join(COLORS[c] for c in tracker.colors)
    # 마지막 상태가 처음 상태와 같은가 (큐브 통째 회전은 같은 것으로 본다)
    target = args.state
    if latest:
        target = VIDEO_GT[vid][0]  # latest·latest2 의 정답 구간은 모두 처음 상태로 끝난다
    elif args.segment in ("AB", "all"):
        target = apply(SEXY_MOVE_STATE, SEXY_MOVE_FINAL)
    elif gt and args.segment == "A":
        target = apply(SEXY_MOVE_STATE, "U")
    elif gt and args.segment == "B":
        target = apply(SEXY_MOVE_STATE, SEXY_MOVE_FINAL)
    elif args.segment in ("Z1", "Z1b", "Z2"):
        target = apply(SEXY_MOVE_STATE, "U")  # Z1·Z1b는 떨어뜨릴 때 돌아간 U 뒤, Z2는 A를 마친 뒤
    whole = set()
    for a in ["", "x", "x2", "x'", "z", "z'"]:
        for b in ["", "y", "y2", "y'"]:
            alg = " ".join(m for m in (a, b) if m)
            whole.add(apply(target, alg) if alg else target)
    # 놓친 구간: 일치율 0.45 이하가 이어진 토막. 다시 잡기까지 걸린 프레임 수 = 끝난 토막의 길이.
    q = np.array(quality)
    lost_runs, run = [], 0
    for v in q:
        if v <= 0.45:
            run += 1
        elif run:
            lost_runs.append(run)
            run = 0
    res = {
        "video": Path(args.video).name, "tracker": "ts" if args.ts else "py", "stride": args.stride, "seed": args.seed,
        "segment": args.segment, "res": args.res, "map_n": args.map_n,  # 2026-11-18 저녁 전 결과에는 없다(그때는 모두 192 입력, 그대로)
        "frames": len(times), "seconds": round(times[-1] - times[0], 1) if times else 0,
        "detected_moves": len(d), "expected_moves": len(exps[best]),
        "move_acc": None if no_gt else 1 - dists[best] / max(len(exps[best]), 1), "edit_distance": None if no_gt else dists[best],
        "final_state_ok": None if no_gt else final in whole,
        "first_lock_t": next((round(t, 2) for t, v in zip(times, quality) if v > 0.45), None),
        "tracked_ratio": float(np.mean(q > 0.45)) if quality else 0.0,
        "lost_episodes": len(lost_runs), "recover_frames_mean": float(np.mean(lost_runs)) if lost_runs else 0.0,
        "recover_frames_max": int(max(lost_runs)) if lost_runs else 0, "lost_tail": int(run),
        "pan": args.pan, "pan_hz": args.pan_hz, "jump": args.jump, "fast": args.fast, "follow": args.follow, "bank": args.bank,
        "ms_per_frame": 1000 * elapsed / max(len(times), 1),
        "relocks": getattr(tracker, "relocks", None), "local_relocks": getattr(tracker, "local_relocks", None), "inits": getattr(tracker, "inits", None),
        "rejected_turns": getattr(tracker, "rejected_turns", None), "off_rejected": getattr(tracker, "off_rejected", None), "reverts": getattr(tracker, "reverts", None), "opts": args.opts,
        "cover": args.cover and not args.no_cover, "settle": args.settle, "moving_frames": moving_frames,
        "events": [[int(e.center), int(e.quarter_turns), round(times[min(e.frame - 1, len(times) - 1)], 2)]
                   + ([info] if (info := (getattr(tracker, "event_info", None) or [None] * len(tracker.events))[n]) else [])
                   for n, e in enumerate(tracker.events)],
        # 층을 일부러 돌리지 않은 구간(Z1, Z1b, Z2): 정답 회전과 짝지은 것(최장 공통 부분열)을 뺀 판정은 전부 가짜
        "false_moves": (len(tracker.events) - (getattr(tracker, "reverts", 0) or 0) - lcs(det, gt)
                        if args.segment in ("Z1", "Z1b", "Z2") or (latest and gt == []) else None),
        "true_found": lcs(det, gt) if args.segment in ("Z1", "Z1b") else None,
        "net_moves": len(d),
    }
    if args.traj:
        res["traj_cols"] = ["t", "quality(None=놓침)", "[u, v, z]", "dx", "dy", "blobOff", "speed", "jumpGain", "layer", "bodyDeg"]
        res["traj"] = traj
    (ROOT / "reports" / f"track_real_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("events", "traj", "traj_cols")}, ensure_ascii=False))
    names = "흰빨초노주파"
    print("판정:", " ".join(f"{names[c]}{'' if k == 1 else '2' if k == 2 else chr(39)}" for c, k in d))
    print("정답:", " ".join(f"{names[c]}{'' if k == 1 else '2' if k == 2 else chr(39)}" for c, k in exps[best]))


if __name__ == "__main__":
    main()
