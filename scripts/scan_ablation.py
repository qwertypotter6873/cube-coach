"""처음 상태 읽기(면 정면 스캔) 규칙을 하나씩 바꿔 실제 영상 3개에서 잰다. README 5.3절 표의 근거.

실행: .venv/bin/python scripts/scan_ablation.py            (촬영 영상 data/raw/*.mov 필요, 약 5분)
출력: reports/scan_real_ablation.json

변형 (모델·문턱은 그대로, 규칙만 다름)
  before          고치기 전: 관심영역·가운데 칸을 '큐브로 보이는 칸 전체'로 정함 + 칸마다 다수결 + 가운데 색만 같으면 3프레임에 '읽음'
  blob_only       가장 큰 덩어리만 큐브로 봄 (읽기·관심영역) + 고치기 전 모으기·맞추기
  now             지금 규칙 (src/cubecoach/facescan.py 그대로): 가장 큰 덩어리 + 면 통째 투표 + 후보 조합 맞추기
  reject_straddle now + 격자 바깥 양쪽(±2칸 줄)에 스티커가 이어지면 두 면에 걸친 격자로 보고 버림 (넣지 않은 것)
  axes            가장 큰 덩어리 + 격자 두 축 간격을 따로 다시 맞춤 (넣지 않은 것, 읽기 수만 셈)
맞추기는 앱처럼 0.5초마다 시도한다. 맞게 읽음 = 그 면의 정답 격자(네 방향 가운데 하나)와 9칸이 모두 같음.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from track_video import LATEST2_STATE, LATEST_STATE, SEXY_MOVE_STATE, crop  # noqa: E402

from cubecoach.cube import COLORS, FACES, face  # noqa: E402
from cubecoach.facescan import MATCH, OTHER, FaceScanner, cell_color, cube_blob, follow_roi, peaks, read_face  # noqa: E402
from cubecoach.render3d import N_CLASSES  # noqa: E402
from cubecoach.segnet import CubeSegNet  # noqa: E402

VIDEOS = {"sexy_move": ("data/raw/sexy_move.mov", 15.0, SEXY_MOVE_STATE),
          "latest": ("data/raw/latest.mov", 45.0, LATEST_STATE),
          "latest2": ("data/raw/latest2.mov", 64.0, LATEST2_STATE)}
ROT = (6, 3, 0, 7, 4, 1, 8, 5, 2)


def legacy_follow(cls: np.ndarray, roi):
    """고치기 전 관심영역: 그 밖이 아닌 칸 전체의 5~95% 범위."""
    ys, xs = np.nonzero(cls != OTHER)
    if len(xs) < 30:
        return None
    n = cls.shape[0]
    x0, x1 = np.percentile(xs, [5, 95]) / n
    y0, y1 = np.percentile(ys, [5, 95]) / n
    side = max(x1 - x0, y1 - y0) * 2.0
    return (roi[0] + ((x0 + x1) / 2 - 0.5) * roi[2], roi[1] + ((y0 + y1) / 2 - 0.5) * roi[2], float(np.clip(side * roi[2], 0.15, 1.2)))


def read_variant(cls: np.ndarray, heat: np.ndarray, blob: bool, axes: bool = False, straddle: bool = False) -> list[int] | None:
    """read_face 와 같은 흐름에 스위치를 단 것. blob=True, axes=straddle=False 이면 read_face 와 같다."""
    P = peaks(heat, cls)
    m = cls < 7
    if blob:
        m = cube_blob(cls)
        if len(P):
            ij = np.clip(P.astype(int), 0, cls.shape[0] - 1)
            P = P[m[ij[:, 1], ij[:, 0]]]
    if len(P) < 9:
        return None
    cube = np.argwhere(m)
    if len(cube) < 20:
        return None
    cy, cx = cube.mean(0) + 0.5
    c = P[np.argmin(((P - [cx, cy]) ** 2).sum(1))]
    d = np.sqrt(((P - c) ** 2).sum(1))
    near = P[d > 1e-6]
    dn = np.sqrt(((near - c) ** 2).sum(1))
    if len(near) < 4:
        return None
    spacing = float(np.sort(dn)[:4].mean())
    if spacing < 1.5:
        return None
    u = near[np.argmin(dn)] - c
    ang = (np.arctan2(u[1], u[0]) + np.pi / 4) % (np.pi / 2) - np.pi / 4
    u = np.array([np.cos(ang), np.sin(ang)]) * spacing
    v = np.array([-u[1], u[0]])

    def match(u, v):
        ks, err = [], 0.0
        sc = np.sqrt(np.linalg.norm(u) * np.linalg.norm(v))
        for r in (-1, 0, 1):
            for q in (-1, 0, 1):
                dd = np.sqrt(((P - (c + q * u + r * v)) ** 2).sum(1))
                k = int(np.argmin(dd))
                ks.append(k)
                err = max(err, dd[k] / sc)
        return ks, err

    ks, err = match(u, v)
    if axes:  # 맞은 봉우리로 두 축을 따로 다시 잡는다 (두 번까지, 나아질 때만)
        G = np.array([(q, r) for r in (-1, 0, 1) for q in (-1, 0, 1)], float)
        for _ in range(2):
            sol = np.linalg.lstsq(G, P[ks] - c, rcond=None)[0]
            ru, rv = np.linalg.norm(sol[0]), np.linalg.norm(sol[1])
            if min(ru, rv) < 1.0 or max(ru, rv) > 2.5 * min(ru, rv):
                break
            ks2, err2 = match(sol[0], sol[1])
            if err2 >= err:
                break
            u, v, ks, err = sol[0], sol[1], ks2, err2
    if err > MATCH:
        return None
    if axes and len(set(ks)) < 9:
        return None
    if straddle:  # 마주 보는 두 옆면은 한꺼번에 보일 수 없다
        sc = np.sqrt(np.linalg.norm(u) * np.linalg.norm(v))

        def side(a: int, ax: str) -> int:
            pts = [c + a * u + t * v if ax == "u" else c + t * u + a * v for t in (-1, 0, 1)]
            return sum(np.sqrt(((P - p) ** 2).sum(1)).min() <= 0.5 * sc for p in pts)

        if any(side(2, ax) >= 2 and side(-2, ax) >= 2 for ax in ("u", "v")):
            return None
    out = []
    for k in ks:
        col = cell_color(cls, *P[k])
        if col < 0:
            return None
        out.append(col)
    return out


class LegacyScanner(FaceScanner):
    """고치기 전 모으기·맞추기: 가운데 색만 같으면 3프레임에 '읽음', 칸 다수결 + 자신 없는 칸 1~2개 바꾸기."""

    def ready(self, need: int = 3) -> bool:
        return bool((self.frames >= need).all())

    def solve(self) -> str | None:
        grids = self.votes.argmax(2)
        conf = np.sort(self.votes, 2)
        margin = (conf[..., -1] + 1) / (conf[..., -2] + 1)
        order = sorted((margin[f, i], f, i) for f in range(6) for i in range(9) if i != 4)
        second = np.argsort(self.votes, 2)[..., -2]
        cands = [grids]
        for m in range(1, 3):
            for combo in itertools.combinations(order[:8], m):
                g = grids.copy()
                for _, f, i in combo:
                    g[f, i] = second[f, i]
                cands.append(g)
        for g in cands:
            sols = self._orient(g)
            if len(sols) == 1:
                return sols[0]
        return None


class Run:
    """변형 하나: 읽기 결과를 받아 맞게/틀리게 센 뒤 스캐너에 넣고, 0.5초마다 맞춰 본다."""

    def __init__(self, scanner: FaceScanner | None, truth_faces: dict[int, set]):
        self.sc, self.tf = scanner, truth_faces
        self.good, self.bad = [0] * 6, [0] * 6
        self.solved, self.last = None, -1.0

    def step(self, t: float, f: list[int] | None, truth: str) -> None:
        if f:
            c = f[4]
            if tuple(f) in self.tf[c]:
                self.good[c] += 1
            else:
                self.bad[c] += 1
            if self.sc is not None and self.solved is None:
                self.sc.add(f)
        if self.sc is not None and self.solved is None and self.sc.ready() and t - self.last >= 0.5:
            self.last = t
            s = self.sc.solve()
            if s:
                self.solved = {"t": round(t, 2), "정답과_같음": s == truth}

    def result(self) -> dict:
        names = "흰빨초노주파"
        return {"맞추기": self.solved, "맞게_읽은_프레임": dict(zip(names, self.good)), "틀리게_읽은_프레임": dict(zip(names, self.bad)),
                "맞게_합": sum(self.good), "틀리게_합": sum(self.bad)}


def truth_faces(state: str) -> dict[int, set]:
    out = {}
    for s in FACES:
        g = face(state, s)
        rs = [tuple(COLORS.index(x) for x in g)]
        for _ in range(3):
            rs.append(tuple(rs[-1][i] for i in ROT))
        out[COLORS.index(g[4])] = set(rs)
    return out


def main() -> None:
    m = CubeSegNet()
    m.load_state_dict(torch.load(ROOT / "models" / "cube_seg_net.pt", map_location="cpu", weights_only=True))
    m.eval()
    out = {"설명": (__doc__ or "").strip().splitlines()[0], "변형": {}}
    for vid, (path, end, truth) in VIDEOS.items():
        tf = truth_faces(truth)
        res = {}
        for follow_kind in ("legacy", "blob"):
            if follow_kind == "legacy":
                runs = {"before": (Run(LegacyScanner(), tf), lambda cls, heat: read_variant(cls, heat, blob=False))}
            else:
                runs = {"blob_only": (Run(LegacyScanner(), tf), lambda cls, heat: read_face(cls, heat)),
                        "now": (Run(FaceScanner(), tf), lambda cls, heat: read_face(cls, heat)),
                        "reject_straddle": (Run(FaceScanner(), tf), lambda cls, heat: read_variant(cls, heat, blob=True, straddle=True)),
                        "axes": (Run(None, tf), lambda cls, heat: read_variant(cls, heat, blob=True, axes=True))}
            cap = cv2.VideoCapture(str(ROOT / path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            roi, i = None, 0
            while True:
                ok, fr = cap.read()
                t = i / fps
                i += 1
                if not ok or t > end:
                    break
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                r = roi or (0.0, 0.0, rgb.shape[0] / rgb.shape[1])
                with torch.no_grad():
                    o = m(torch.from_numpy(crop(rgb, r)).permute(2, 0, 1)[None].float() / 255)[0]
                cls = o[:N_CLASSES].argmax(0).numpy()
                heat = torch.sigmoid(o[N_CLASSES]).numpy()
                for run, read in runs.values():
                    run.step(t, read(cls, heat), truth)
                roi = legacy_follow(cls, r) if follow_kind == "legacy" else follow_roi(cls, r)
            for k, (run, _) in runs.items():
                res[k] = run.result()
                print(vid, k, res[k]["맞추기"], "맞게", res[k]["맞게_합"], "틀리게", res[k]["틀리게_합"], flush=True)
        out["변형"][vid] = {"영상": path, "끝": end, **res}
    p = ROOT / "reports" / "scan_real_ablation.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(p)


if __name__ == "__main__":
    main()
