"""처음 상태 읽기: 한 면씩 정면으로 보여 주면 3x3 격자를 읽고, 여섯 면을 모아 상태를 만든다.

1. 면 읽기 (프레임마다): 큐브로 보이는 칸(스티커·몸체)이 이어진 덩어리 가운데 가장 큰 것만 큐브로 본다
   (뒤쪽 책장처럼 큐브와 떨어진 곳이 스티커로 읽혀도 가운데·관심영역이 끌려가지 않게). 그 덩어리 안의
   중심 지도 봉우리 가운데 덩어리 한가운데에 가장 가까운 것을 가운데 칸으로 보고, 가까운 이웃으로
   격자 방향·간격을 잡아 9칸 자리를 예측한다. 9자리 모두에 봉우리가 있으면 '정면으로 보인 면'으로 인정한다.
   칸 색은 칸 자리 둘레 3x3 지도 칸의 다수결로 정한다 (흰 센터의 파란 로고처럼 한가운데만 다른 색인 경우를
   견디도록). 화면에서 보이는 그대로(위→아래, 왼→오) 읽는다.
2. 모으기: 가운데 색마다 (a) 9칸의 색을 프레임에 걸쳐 투표하고, (b) 9칸 통째(회전 무시)로 같은 격자가
   몇 번 나왔는지도 센다. 기울여 든 면에서는 두 면에 걸친 격자처럼 틀린 읽기가 몇 프레임씩 같은 모양으로
   나와서, 칸 다수결만 쓰면 드물게 맞게 읽힌 면이 틀린 읽기에 덮인다.
3. 맞추기: 면마다 후보 격자(통째로 NEED번 이상 나온 격자, 많이 나온 순으로 3개 + 칸 다수결 격자)를 두고,
   1등끼리 → 한 면만 다른 후보로 바꾼 조합 순으로 맞춰 본다. 조합마다 각 면을 0/90/180/270도 돌린
   4^6 = 4,096가지 가운데 실제로 존재할 수 있는(cube.validate 통과) 서로 다른 상태가 정확히 하나인지 본다
   (다 맞춘 면처럼 돌려도 같은 면이 있으면 여러 방향 조합이 같은 상태를 만들므로, 같은 상태는 하나로 센다).
   사람이 면을 어느 방향으로 들고 보여 줬는지 몰라도 된다. 그래도 안 맞으면 칸 다수결 격자에서
   가장 자신 없는 칸들을 둘째 후보 색으로 바꿔 본다.

브라우저판: web/src/vision/facescan.ts (같은 규칙)
"""

from __future__ import annotations

import itertools

import cv2
import numpy as np

from .cube import COLORS, FACES, rotate_face_cw, validate

STICKER = 6  # 분류 0~5가 스티커 색
PEAK_MIN = 0.35
MATCH = 0.38  # 예측한 칸 자리와 봉우리의 거리 한도 (칸 간격 대비)
SLOT_COLOR = {"U": 0, "R": 1, "F": 2, "D": 3, "L": 4, "B": 5}  # 표준 배색: 가운데 색 → 면 자리
NEED = 3  # 면 하나에 같은 격자(회전 무시)가 이만큼 나와야 맞추기 후보로 쓴다 (ready 의 프레임 수와 같은 값)
OTHER = 7
_ROT = (6, 3, 0, 7, 4, 1, 8, 5, 2)  # 3x3 격자를 시계 방향으로 90도


def cube_blob(cls: np.ndarray) -> np.ndarray:
    """큐브로 보이는 칸(그 밖이 아닌 칸)이 8방향으로 이어진 덩어리 가운데 가장 큰 것 (bool 지도)."""
    m = (cls != OTHER).astype(np.uint8)
    k, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if k <= 1:
        return m.astype(bool)
    return lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))


def follow_roi(cls: np.ndarray, roi):
    """큐브로 보이는 곳(가장 큰 덩어리)을 따라가는 다음 관심영역 (큐브가 영역의 절반쯤 차게)."""
    ys, xs = np.nonzero(cube_blob(cls))
    if len(xs) < 30:
        return None
    n = cls.shape[0]
    x0, x1 = np.percentile(xs, [5, 95]) / n
    y0, y1 = np.percentile(ys, [5, 95]) / n
    side = max(x1 - x0, y1 - y0) * 2.0
    cx, cy = (x0 + x1) / 2 - 0.5, (y0 + y1) / 2 - 0.5
    return (roi[0] + cx * roi[2], roi[1] + cy * roi[2], float(np.clip(side * roi[2], 0.15, 1.2)))


def _canon(g) -> tuple:
    """격자를 회전과 상관없이 같은 것끼리 묶는 열쇠 (네 방향 가운데 사전순 최소)."""
    rs = [tuple(int(v) for v in g)]
    for _ in range(3):
        rs.append(tuple(rs[-1][i] for i in _ROT))
    return min(rs)


def peaks(heat: np.ndarray, cls: np.ndarray) -> np.ndarray:
    n = heat.shape[0]
    pad = np.pad(heat, 1)
    neigh = np.stack([pad[1 + dy:1 + dy + n, 1 + dx:1 + dx + n] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    ok = (heat >= neigh.max(0)) & (heat > PEAK_MIN) & (cls < STICKER)
    ys, xs = np.nonzero(ok)
    w = neigh[:, ys, xs]
    off = np.array([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dtype=np.float64)
    return np.stack([xs, ys], 1) + 0.5 + (w.T @ off) / np.maximum(w.sum(0), 1e-6)[:, None]


def cell_color(cls: np.ndarray, x: float, y: float) -> int:
    """칸 자리 둘레 3x3 지도 칸에서 가장 많은 스티커 색 (없으면 -1)."""
    n = cls.shape[0]
    i, j = int(y), int(x)
    win = cls[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].ravel()
    win = win[win < STICKER]
    return int(np.bincount(win, minlength=6).argmax()) if len(win) else -1


def read_face(cls: np.ndarray, heat: np.ndarray) -> list[int] | None:
    """정면으로 보이는 면 하나의 9칸 색 (화면에서 보이는 그대로, 행 우선). 못 읽으면 None."""
    P = peaks(heat, cls)
    blob = cube_blob(cls)
    if len(P):
        n = cls.shape[0]
        ij = np.clip(P.astype(int), 0, n - 1)
        P = P[blob[ij[:, 1], ij[:, 0]]]  # 가장 큰 덩어리 안의 봉우리만
    if len(P) < 9:
        return None
    cube = np.argwhere(blob)
    if len(cube) < 20:
        return None
    cy, cx = cube.mean(0) + 0.5
    c = P[np.argmin(((P - [cx, cy]) ** 2).sum(1))]
    d = np.sqrt(((P - c) ** 2).sum(1))
    near = P[(d > 1e-6)]
    dn = np.sqrt(((near - c) ** 2).sum(1))
    if len(near) < 4:
        return None
    spacing = float(np.sort(dn)[:4].mean())
    if spacing < 1.5:
        return None
    # 격자 방향: 가장 가까운 이웃 쪽 벡터를 -45~45도(오른쪽)로 돌려 맞춘다
    u = near[np.argmin(dn)] - c
    ang = np.arctan2(u[1], u[0])
    ang = (ang + np.pi / 4) % (np.pi / 2) - np.pi / 4
    u = np.array([np.cos(ang), np.sin(ang)]) * spacing
    v = np.array([-u[1], u[0]])  # 화면 y가 아래쪽이므로 오른쪽을 시계 방향으로 90도 = 아래쪽
    out = []
    for r in (-1, 0, 1):
        for q in (-1, 0, 1):
            p = c + q * u + r * v
            dd = np.sqrt(((P - p) ** 2).sum(1))
            k = int(np.argmin(dd))
            if dd[k] > MATCH * spacing:
                return None
            col = cell_color(cls, *P[k])
            if col < 0:
                return None
            out.append(col)
    return out


class FaceScanner:
    def __init__(self) -> None:
        self.votes = np.zeros((6, 9, 6), dtype=np.float64)  # 가운데 색 x 칸 x 색
        self.frames = np.zeros(6, dtype=int)
        self.grids: list[dict[tuple, list]] = [{} for _ in range(6)]  # 가운데 색마다 {회전 무시 열쇠: [격자, 횟수]}
        # 마지막 solve()에서 1등 격자끼리 맞춰 보니 서로 다른 유효한 상태가 둘 이상이었다 (면만으로는 가릴 수 없는 상태)
        self.ambiguous = False

    def add(self, face: list[int]) -> int:
        """읽은 면 하나를 넣는다. 가운데 색을 돌려준다. 같은 면은 회전을 맞춰서 더한다."""
        center = face[4]
        g = list(face)
        if self.frames[center]:
            # 지난번과 같은 방향이 되도록 0/90/180/270도 가운데 가장 잘 맞는 것으로
            ref = self.votes[center].argmax(1)
            best, bg = -1, g
            for k in range(4):
                s = sum(int(a == b) for a, b in zip(g, ref))
                if s > best:
                    best, bg = s, list(g)
                g = [g[i] for i in (6, 3, 0, 7, 4, 1, 8, 5, 2)]
            g = bg
        for i, col in enumerate(g):
            self.votes[center, i, col] += 1
        self.frames[center] += 1
        e = self.grids[center].setdefault(_canon(face), [list(face), 0])
        e[1] += 1
        return center

    def agree(self, center: int) -> int:
        """그 면에서 가장 많이 나온 격자(회전 무시)의 횟수."""
        return max((e[1] for e in self.grids[center].values()), default=0)

    def weakest(self) -> int:
        """가장 덜 확실한 면의 가운데 색 (같은 격자가 가장 적게 나온 면). 다시 보여 달라고 할 때 쓴다."""
        return min(range(6), key=lambda c: (self.agree(c), self.frames[c]))

    def ready(self, need: int = NEED) -> bool:
        """여섯 면 모두 같은 격자가 need번 이상 나왔다 (가운데 색만 맞는 틀린 읽기로는 차지 않는다)."""
        return all(self.agree(c) >= need for c in range(6))

    def candidates(self) -> list[list[np.ndarray]]:
        """면마다 후보 격자: 통째로 NEED번 이상 나온 격자(많은 순 3개) + 칸 다수결 격자. 비면 칸 다수결만."""
        cell = self.votes.argmax(2)
        out = []
        for c in range(6):
            whole = sorted(self.grids[c].values(), key=lambda e: -e[1])
            lst = [np.array(g) for g, n in whole if n >= NEED][:3]
            if _canon(cell[c]) not in {_canon(g) for g in lst}:
                lst.append(cell[c])
            out.append(lst)
        return out

    def solve(self) -> str | None:
        self.ambiguous = False
        cands = self.candidates()
        first = np.stack([lst[0] for lst in cands])
        sols = self._orient(first)
        if len(sols) == 1:
            return sols[0]
        if len(sols) > 1:  # 다른 후보로 억지로 맞추지 않는다 (브라우저판 FaceScanner.ambiguous 와 같음)
            self.ambiguous = True
            return None
        tries = []  # 한 면만 2·3등(또는 칸 다수결)으로 바꾼 조합
        for f in range(6):
            for g in cands[f][1:]:
                t = first.copy()
                t[f] = g
                tries.append(t)
        for g in tries:
            sols = self._orient(g)
            if len(sols) == 1:
                return sols[0]
        # 칸 다수결 격자에서 자신 없는 칸을 둘째 후보 색으로 바꿔 본다
        grids = self.votes.argmax(2)  # (6, 9)
        conf = np.sort(self.votes, 2)
        margin = (conf[..., -1] + 1) / (conf[..., -2] + 1)
        cands = [grids]
        # 자신 없는 칸(가운데 제외) 몇 개를 둘째 후보로 바꾼 것도 견준다
        order = [(margin[f, i], f, i) for f in range(6) for i in range(9) if i != 4]
        order.sort()
        second = np.argsort(self.votes, 2)[..., -2]
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

    @staticmethod
    def _orient(grids: np.ndarray) -> list[str]:
        if (np.bincount(np.asarray(grids).ravel(), minlength=6) != 9).any():
            return []  # 색마다 9칸이 아니면 어느 방향으로 돌려도 안 맞는다
        faces = {}
        for slot in FACES:
            s = "".join(COLORS[c] for c in grids[SLOT_COLOR[slot]])
            faces[slot] = [s]
            for _ in range(3):
                faces[slot].append(rotate_face_cw(faces[slot][-1]))
        sols: dict[str, None] = {}  # 서로 다른 상태만 (돌려도 같은 면이 있으면 여러 방향 조합이 같은 상태를 만든다)
        for ks in itertools.product(range(4), repeat=6):
            st = "".join(faces[slot][k] for slot, k in zip(FACES, ks))
            if st not in sols and not validate(st):
                sols[st] = None
        return list(sols)
