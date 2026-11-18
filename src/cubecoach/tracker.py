"""상태를 아는 큐브를 매 프레임 끊김 없이 추적한다: 자세 + 돌아가는 층의 각도.

인식 모델(segnet)이 내는 두 지도만 입력으로 쓴다.
  logp (8, H, W): 칸마다 [스티커 색 6 / 큐브 몸체 / 그 밖]의 로그확률
  heat (H, W)   : 스티커 중심일 가능성 0~1

추적하는 것
  R, T        큐브 몸통의 자세 (회전·위치)
  layer, phi  지금 돌아가고 있는 층과 그 각도 (없으면 layer = -1)

점수 (가설이 화면을 얼마나 잘 설명하는가) — 양방향으로 따진다
  가설 → 화면: 54칸을 가설대로 투영했을 때 각 칸 자리에 기대한 색이 있는가.
               손에 가린 칸("그 밖")은 거의 중립이라 보이는 칸만으로 맞춰진다.
  화면 → 가설: 화면에서 찾은 스티커 하나하나가 가설의 어느 칸으로 설명되는가.
               이 항이 없으면 안 맞는 면을 안 보이는 쪽으로 기울여 숨기는 가설이 이긴다.

흐름
- 매 프레임 가설을 조금씩 흔들어 점수가 가장 높은 것으로 옮긴다.
- 층이 돌기 시작하면 그 층의 각도를 함께 따라간다. 도는 도중에도 54칸 전부가
  설명되므로 자세가 움직이는 층에 끌려가지 않는다.
- 각도가 90도 단위에 멈추면, 가능한 회전 후보들을 화면과 대조해 회전을 확정한다.
- 따라간 회전이 없는데 안 맞는 칸이 계속 보이면, 보지 못한 회전이 있었다고 보고 찾는다.

회전은 "어느 가운데 색의 층을 시계 방향으로 몇 번(90도 단위) 돌렸는가"로 보고한다.
가운데 색 기준이라서 큐브를 어느 방향으로 들고 있든 같은 뜻이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cube import FACES, FRAMES, MOVES
from .geometry import CENTERS, NORMALS, random_rotation, rotvec
from .render3d import BODY, OTHER

DEG = np.pi / 180
AXES = np.array([FRAMES[f][0] for f in FACES], dtype=np.float64)  # 면마다 바깥쪽 법선
LAYER = ((CENTERS - 0.5 * NORMALS) @ AXES.T > 0.5).T  # (6, 54): 그 면의 층에 속한 칸

# 가설 → 화면
P0 = 0.2  # 이보다 확률이 높으면 "맞는다"는 증거로 본다
WRONG, ON_BODY, OCCLUDED, OUTSIDE = -3.0, -1.2, -0.4, -1.0
MATCH_MAX = 2.0
HEAT_BONUS = 1.5
# 화면 → 가설
PEAK_MIN, PEAK_MAX_N = 0.35, 48
UNEXPLAINED = 2.0  # 설명되지 않는 스티커 하나의 벌점
PEAK_SIGMA = 1.3  # 지도 칸 단위: 이 거리 안에 같은 색 칸이 투영되면 설명된 것으로 본다

ROI_SCALE = 2.8  # 관심영역 한 변 = 정면 한 면 크기의 2.8배
# 한 면만 정면으로 보이면 기울기는 화면만으로 정해지지 않는다. 같은 점수라면
# 직전 자세에 가까운 쪽을 고르도록 약한 벌점을 준다 (10도 = 1점, 스티커 한 칸 이동 = 3점).
INERTIA_ROT, INERTIA_T, INERTIA_S = 1.0 / 10**2, 3.0, 0.5 / 0.05**2

TURN_START_GAIN = 3.0  # "층이 돌기 시작했다"는 가설이 이만큼 앞서야 받아들인다
SNAP, CANCEL = 14 * DEG, 8 * DEG
DOUBT_QUALITY, DOUBT_FRAMES = 0.92, 3  # 돌리는 중이 아닌데 안 맞는 칸이 계속 보이면 상태를 의심한다
RECOVER_MARGIN, RECOVER_LEAD, TIE = 5.0, 1.0, 2.0
HISTORY = 3  # 놓친 회전을 최근 몇 번의 회전 앞에까지 끼워 넣어 볼지
LOST_QUALITY, LOST_FRAMES = 0.45, 10

_ROTATION_OF = {"U": "y", "D": "y'", "R": "x", "L": "x'", "F": "z", "B": "z'"}  # 그 면과 같은 방향의 큐브 전체 회전


def _power(sym: str, k: int) -> str:
    """기호 sym을 k번(90도 단위, 음수면 반대로) 적용한 기호. 예: ("y'", -1) → "y"."""
    base, sign = sym.rstrip("'"), -1 if sym.endswith("'") else 1
    return {1: base, 2: base + "2", 3: base + "'"}[(sign * k) % 4]


def _perm(alg: str) -> np.ndarray:
    p = np.arange(54)
    for m in alg.split():
        p = p[MOVES[m]]
    return p


def _candidates() -> tuple[np.ndarray, list[tuple[int, int]]]:
    """회전을 확정할 때 견줄 후보: 18가지 회전 x 2가지 해석.

    같은 회전이라도 추적하던 자세가 "두 층짜리 몸통"을 따라갔는지 "돌아간 한 층"을
    따라갔는지에 따라 칸 배치가 달리 보인다. 뒤의 경우는 회전한 다음 큐브 전체를
    같은 축으로 그만큼 되돌린 배치와 같다 (예: U 다음 y').
    """
    perms, events = [], []
    for face in FACES:
        for k in (1, 2, 3):
            move = _power(face, k)
            for alg in (move, f"{move} {_power(_ROTATION_OF[face], -k)}"):
                perms.append(_perm(alg))
                events.append((FACES.index(face), k))
    return np.array(perms), events


CAND_PERMS, CAND_EVENTS = _candidates()


@dataclass
class Event:
    center: int  # 돌린 층의 가운데 색 번호
    quarter_turns: int  # 시계 방향 90도 단위 (3 = 반시계 90도)
    frame: int


class Tracker:
    def __init__(self, colors: np.ndarray, f: float = 0.95, seed: int = 0) -> None:
        self.colors = np.asarray(colors).copy()
        self.f = f
        self.R: np.ndarray | None = None
        self.T: np.ndarray | None = None
        self.layer, self.phi = -1, 0.0
        self.rng = np.random.default_rng(seed)
        self.bank = random_rotation(np.random.default_rng(123), 2500)
        self.still = self.idle = self.bad_frames = self.doubt = 0
        self.frame = 0
        self.quality = 0.0
        self.events: list[Event] = []
        self.log: list[tuple[np.ndarray, np.ndarray]] = []  # 최근 회전들: (회전 전 상태, 적용한 순열)
        self._peaks = (np.zeros((0, 2)), np.zeros(0, dtype=int), np.zeros(0))

    # --- 관심영역 ---------------------------------------------------------
    def roi(self) -> tuple[float, float, float]:
        """다음 프레임에서 인식 모델에 넣을 정사각 영역 (프레임 너비 기준)."""
        if self.T is None:
            return (0.0, 0.0, 1.0)
        unit = self.f / self.T[2]
        return (float(self.f * self.T[0] / self.T[2]), float(self.f * self.T[1] / self.T[2]), float(3 * unit * ROI_SCALE))

    # --- 화면에서 스티커 찾기 ----------------------------------------------
    def _observe(self, logp: np.ndarray, heat: np.ndarray) -> None:
        """중심 지도의 봉우리 = 화면에 보이는 스티커들. 위치(칸 단위)·색·세기를 기억해 둔다."""
        pad = np.pad(heat, 1)
        n = heat.shape[0]
        neigh = np.stack([pad[1 + dy : 1 + dy + n, 1 + dx : 1 + dx + n] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        is_peak = (heat >= neigh.max(0)) & (heat > PEAK_MIN) & (np.exp(logp[:6]).sum(0) > 0.5)
        ys, xs = np.nonzero(is_peak)
        if len(ys) > PEAK_MAX_N:
            keep = np.argsort(-heat[ys, xs])[:PEAK_MAX_N]
            ys, xs = ys[keep], xs[keep]
        # 주변 3x3의 무게중심으로 칸보다 세밀한 위치를 얻는다
        wts = neigh[:, ys, xs]
        off = np.array([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dtype=np.float64)
        xy = np.stack([xs, ys], axis=1) + 0.5 + (wts.T @ off) / np.maximum(wts.sum(0), 1e-6)[:, None]
        self._peaks = (xy, logp[:6, ys, xs].argmax(0), heat[ys, xs])

    # --- 투영과 점수 ------------------------------------------------------
    def _project(self, Rs, Ts, layers, phis, roi, n: int):
        """가설 N개 각각에서 54칸 중심의 지도 좌표 (N, 54, 2)와 신뢰 가중치 (N, 54)."""
        N = len(Rs)
        C = np.broadcast_to(CENTERS, (N, 54, 3))
        Nm = np.broadcast_to(NORMALS, (N, 54, 3))
        turning = layers >= 0
        if turning.any():
            M = rotvec(-phis[:, None] * AXES[np.maximum(layers, 0)])  # 바깥에서 봐서 시계 방향이 +
            in_layer = (LAYER[np.maximum(layers, 0)] & turning[:, None])[..., None]
            C = np.where(in_layer, np.einsum("nij,kj->nki", M, CENTERS), C)
            Nm = np.where(in_layer, np.einsum("nij,kj->nki", M, NORMALS), Nm)
        Xc = np.einsum("nij,nkj->nki", Rs, C) + Ts[:, None, :]
        nc = np.einsum("nij,nkj->nki", Rs, Nm)
        facing = -(nc * Xc).sum(2) / np.linalg.norm(Xc, axis=2)
        w = np.clip((facing - 0.12) / 0.25, 0, 1)  # 비스듬히 보이는 면은 덜 믿는다
        uv = self.f * Xc[..., :2] / Xc[..., 2:3]
        px = ((uv - np.array(roi[:2])) / roi[2] + 0.5) * n
        return px, w

    @staticmethod
    def _lookup(px, heat):
        n = heat.shape[0]
        xi, yi = np.floor(px[..., 0]).astype(int), np.floor(px[..., 1]).astype(int)
        inside = (xi >= 0) & (xi < n) & (yi >= 0) & (yi < n)
        xi, yi = np.clip(xi, 0, n - 1), np.clip(yi, 0, n - 1)
        # 중심 지도는 쌍선형 보간으로 읽는다 (자세를 칸보다 세밀하게 맞추기 위해)
        fx, fy = px[..., 0] - 0.5, px[..., 1] - 0.5
        x0, y0 = np.clip(np.floor(fx).astype(int), 0, n - 2), np.clip(np.floor(fy).astype(int), 0, n - 2)
        ax, ay = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
        h = (heat[y0, x0] * (1 - ax) + heat[y0, x0 + 1] * ax) * (1 - ay) + (heat[y0 + 1, x0] * (1 - ax) + heat[y0 + 1, x0 + 1] * ax) * ay
        return xi, yi, inside, h

    def score(self, Rs, Ts, layers, phis, logp, heat, roi, colors: np.ndarray | None = None) -> np.ndarray:
        """가설 N개의 점수 (N,). colors를 (M, 54)로 주면 상태 M개에 대해 (N, M)."""
        px, w = self._project(Rs, Ts, layers, phis, roi, heat.shape[0])
        xi, yi, inside, h = self._lookup(px, heat)
        states = self.colors[None] if colors is None else colors

        # 가설 → 화면
        lp = logp[:, yi, xi].transpose(1, 2, 0)  # (N, 54, 8)
        cls = lp.argmax(2)
        expected = np.take_along_axis(lp[:, None], states[None, :, :, None], axis=3)[..., 0]  # (N, M, 54)
        term = np.clip(expected - np.log(P0), WRONG, MATCH_MAX)
        term = np.where((cls == BODY)[:, None, :], np.maximum(term, ON_BODY), term)
        term = term + HEAT_BONUS * h[:, None, :]
        term = np.where((cls == OTHER)[:, None, :], OCCLUDED, term)
        term = np.where(inside[:, None, :], term, OUTSIDE)
        s = (w[:, None, :] * term).sum(2)

        # 화면 → 가설: 찾은 스티커마다, 같은 색으로 투영된 가장 가까운 칸까지의 거리
        xy, pcol, pw = self._peaks
        if len(xy):
            d2 = ((px[:, None, :, :] - xy[None, :, None, :]) ** 2).sum(-1)  # (N, P, 54)
            d2 = np.where((w > 0.05)[:, None, :], d2, np.inf)
            for m in range(len(states)):
                same = states[m][None, :] == pcol[:, None]  # (P, 54)
                nearest = np.where(same[None], d2, np.inf).min(2)  # (N, P)
                s[:, m] -= UNEXPLAINED * ((1 - np.exp(-nearest / (2 * PEAK_SIGMA**2))) * pw).sum(1)
        return s[:, 0] if colors is None else s

    def _quality(self, logp, heat, roi) -> float:
        """보이는 칸 가운데 기대 색과 일치하는 비율."""
        px, w = self._project(self.R[None], self.T[None], np.array([self.layer]), np.array([self.phi]), roi, heat.shape[0])
        xi, yi, inside, _ = self._lookup(px, heat)
        cls = logp[:, yi[0], xi[0]].argmax(0)
        seen = inside[0] & (cls != OTHER)
        denom = float((w[0] * seen).sum())
        return float((w[0] * seen * (cls == self.colors)).sum() / denom) if denom > 1.5 else 0.0

    # --- 탐색 ------------------------------------------------------------
    def _perturb(self, R, T, n, sig_deg, sig_t, sig_s):
        Rs = rotvec(self.rng.normal(0, sig_deg * DEG, (n, 3))) @ R
        Ts = T + np.concatenate([self.rng.normal(0, sig_t, (n, 2)), np.zeros((n, 1))], axis=1)
        Ts = Ts * np.exp(self.rng.normal(0, sig_s, (n, 1)))  # 시선 방향으로 당기고 밀기 = 크기 변화
        Rs[0], Ts[0] = R, T  # 지금 값도 후보에 둔다
        return Rs, Ts

    @staticmethod
    def _inertia(R0, T0, Rs, Ts) -> np.ndarray:
        angle = np.degrees(np.arccos(np.clip((np.einsum("ij,nij->n", R0, Rs) - 1) / 2, -1, 1)))
        scale = np.log(Ts[:, 2] / T0[2])
        shift = np.linalg.norm(Ts[:, :2] / Ts[:, 2:3] - T0[:2] / T0[2], axis=1) * T0[2]
        return INERTIA_ROT * angle**2 + INERTIA_T * shift**2 + INERTIA_S * scale**2

    def _sweep_angle(self, logp, heat, roi) -> None:
        """돌아가는 층의 각도를 넓게 훑는다.

        층은 90도마다 같은 모양이라, 조금씩만 흔들면 실제보다 90도 뒤처진 닮은꼴 위치에
        갇힐 수 있다. 각도는 한 방향 값이므로 매 프레임 촘촘히 다 넣어 본다.
        "층이 돈 경우"와 "층은 그대로고 몸통이 반대로 돈 경우"를 모두 넣는다.
        """
        d = np.arange(-66, 67, 6) * DEG
        phis = np.concatenate([self.phi + d, self.phi + d])
        Rs = np.repeat(self.R[None], len(phis), axis=0)
        Rs[len(d) :] = self.R @ rotvec(d[:, None] * AXES[self.layer])
        Ts = np.repeat(self.T[None], len(phis), axis=0)
        s = self.score(Rs, Ts, np.full(len(phis), self.layer), phis, logp, heat, roi)
        s[len(d) :] -= INERTIA_ROT * np.degrees(d) ** 2 * 0.25
        i = int(s.argmax())
        self.R, self.phi = Rs[i], float(phis[i])

    ROUNDS = ((9.0, 0.2, 0.05, 6.0), (4.0, 0.09, 0.02, 3.0), (1.5, 0.035, 0.008, 1.5), (0.6, 0.015, 0.003, 0.6))

    def _refine(self, logp, heat, roi, rounds=ROUNDS, n: int = 160, inertia: bool = True) -> float:
        """자세(와 돌아가는 층의 각도)를 조금씩 흔들어 가장 잘 맞는 값으로 옮긴다."""
        R0, T0 = self.R, self.T
        if self.layer >= 0:
            self._sweep_angle(logp, heat, roi)
        best = -np.inf
        for sig_deg, sig_t, sig_s, sig_phi in rounds:
            Rs, Ts = self._perturb(self.R, self.T, n, sig_deg, sig_t, sig_s)
            layers = np.full(n, self.layer)
            phis = np.full(n, self.phi)
            if self.layer >= 0:
                phis = phis + self.rng.normal(0, sig_phi * DEG, n)
                phis[0] = self.phi
            s = self.score(Rs, Ts, layers, phis, logp, heat, roi)
            if inertia:
                s = s - self._inertia(R0, T0, Rs, Ts)
            i = int(s.argmax())
            best, self.R, self.T, self.phi = float(s[i]), Rs[i], Ts[i], float(phis[i])
        return best

    def _probe_turn(self, base: float, logp, heat, roi) -> None:
        """여섯 층 가운데 하나가 돌기 시작했는지 떠본다.

        같은 회전이라도 손에 따라 "층이 돈다"와 "층을 잡고 나머지 몸통이 돈다"
        두 가지로 나타나므로 둘 다 가정해 본다 (뒤의 경우는 자세가 함께 변한다).
        """
        angles = np.array([-75, -60, -45, -30, -15, 15, 30, 45, 60, 75]) * DEG
        reps = 2
        layers = np.repeat(np.arange(6), len(angles) * 2 * reps)
        phis = np.tile(np.repeat(angles, 2 * reps), 6)
        held = np.tile(np.repeat([False, True], reps), 6 * len(angles))
        Rs, Ts = self._perturb(self.R, self.T, len(layers), 2.5, 0.05, 0.012)
        Rs[::reps], Ts[::reps] = self.R, self.T
        Rs[held] = Rs[held] @ rotvec(phis[held, None] * AXES[layers[held]])
        s = self.score(Rs, Ts, layers, phis, logp, heat, roi)
        s[~held] -= self._inertia(self.R, self.T, Rs[~held], Ts[~held])
        i = int(s.argmax())
        if s[i] - base > TURN_START_GAIN:
            self.R, self.T, self.layer, self.phi = Rs[i], Ts[i], int(layers[i]), float(phis[i])
            self.still = self.idle = 0

    def _decide(self, logp, heat, roi, tracked: tuple[int, int] | None) -> Event | None:
        """가만히 있는 지금 화면을 가장 잘 설명하는 상태를 고른다.

        tracked = (층, 90도 단위): 층 각도를 따라가다 90도 단위에 멈춘 경우.
        tracked = None: 따라간 회전은 없는데 안 맞는 칸이 계속 보이는 경우. 보지 못한 회전이
        있었다고 보고, 지금 또는 최근 몇 번의 회전 앞에 회전 하나를 끼워 넣은 상태들을 견준다.
        """
        n = 48
        Rs, Ts = self._perturb(self.R, self.T, n, 3.0, 0.06, 0.015)
        blocks = [self.colors[None], self.colors[CAND_PERMS]]
        for back in range(1, (0 if tracked is not None else min(HISTORY, len(self.log))) + 1):
            cand = self.log[-back][0][CAND_PERMS]
            for _, perm in self.log[-back:]:
                cand = cand[:, perm]
            blocks.append(cand)
        table = self.score(Rs, Ts, np.full(n, -1), np.zeros(n), logp, heat, roi, colors=np.concatenate(blocks))
        s = table.max(0)
        order = np.argsort(-s[1:])
        best, second = int(order[0]), int(order[1])
        choice = None
        if tracked is not None:
            t = (tracked[0] * 3 + tracked[1] - 1) * 2
            if best != t and s[1 + best] - s[1 + t] > RECOVER_LEAD:
                choice = best  # 따라간 것보다 화면을 확실히 더 잘 설명하는 회전이 있다
            elif s[1 + t] >= s[0] - TIE:
                choice = t  # 색으로는 구별이 안 되더라도, 층이 도는 것을 봤으므로 믿는다
        elif s[1 + best] - s[0] > RECOVER_MARGIN and s[1 + best] - s[1 + second] > RECOVER_LEAD:
            choice = best
        self.layer, self.phi, self.still = -1, 0.0, 0
        if choice is None:
            return None

        back, j = divmod(choice, len(CAND_PERMS))  # back번의 회전 앞에 끼워 넣는다
        face, k = CAND_EVENTS[j]
        later = [perm for _, perm in self.log[len(self.log) - back :]]
        before = self.log[-back][0] if back else self.colors
        event = Event(int(before[face * 9 + 4]), k, self.frame)
        self.log = self.log[: len(self.log) - back]
        state = before
        for perm in [CAND_PERMS[j]] + later:
            self.log.append((state, perm))
            state = state[perm]
        self.log = self.log[-HISTORY:]
        self.colors = state
        self.events.insert(len(self.events) - back, event)
        i = int(table[:, 1 + choice].argmax())
        self.R, self.T = Rs[i], Ts[i]
        return event

    def _global_init(self, logp, heat, roi) -> bool:
        """처음이거나 놓쳤을 때: 큐브로 보이는 영역에서 위치·크기를 어림하고 모든 방향을 훑는다."""
        n = heat.shape[0]
        mask = np.exp(logp[:7]).sum(0) > 0.5
        if mask.sum() < 12:
            return False
        ys, xs = np.nonzero(mask)
        uv = (np.array([xs.mean() + 0.5, ys.mean() + 0.5]) / n - 0.5) * roi[2] + np.array(roi[:2])
        side = np.sqrt(mask.sum() / 1.35) / n * roi[2]  # 큐브 한 변의 겉보기 길이
        Ts = []
        for s in (0.8, 1.0, 1.25, 1.55):
            tz = self.f / (side * s / 3)
            Ts.append([uv[0] * tz / self.f, uv[1] * tz / self.f, tz])
        Ts = np.repeat(np.array(Ts), len(self.bank), axis=0)
        Rs = np.tile(self.bank, (4, 1, 1))
        self.layer, self.phi = -1, 0.0
        s = np.concatenate([
            self.score(Rs[i : i + 500], Ts[i : i + 500], np.full(len(Rs[i : i + 500]), -1), np.zeros(len(Rs[i : i + 500])), logp, heat, roi)
            for i in range(0, len(Rs), 500)
        ])
        best = (-np.inf, None, None)
        for i in np.argsort(-s)[:6]:
            self.R, self.T = Rs[i], Ts[i]
            sc = self._refine(logp, heat, roi, rounds=((14.0, 0.5, 0.1, 0.0), (7.0, 0.25, 0.05, 0.0)) + self.ROUNDS, inertia=False)
            if sc > best[0]:
                best = (sc, self.R, self.T)
        self.R, self.T = best[1], best[2]
        return True

    # --- 한 프레임 --------------------------------------------------------
    def update(self, logp: np.ndarray, heat: np.ndarray, roi) -> list[Event]:
        self.frame += 1
        self._observe(logp, heat)
        new: list[Event] = []
        if self.R is None:
            if self._global_init(logp, heat, roi):
                self.quality = self._quality(logp, heat, roi)
                if self.quality < 0.5:  # 자신 없는 초기화는 버리고 다음 프레임에 다시
                    self.R = self.T = None
            return new

        base = self._refine(logp, heat, roi)
        if self.layer >= 0:
            k = int(np.round(self.phi / (np.pi / 2)))
            resid = self.phi - k * np.pi / 2
            if k % 4 and abs(resid) < SNAP:
                # 90도 단위에 닿았다. 멈추기를 기다리지 않고 바로 확정을 시도한다: 빠르게 이어 돌리면
                # 멈춘 프레임이 없기 때문이다. 180도를 한 번에 돌리는 중이었다면 같은 층이 계속
                # 도는 것으로 다시 잡혀 90도가 한 번 더 확정된다.
                event = self._decide(logp, heat, roi, (self.layer, k % 4))
                if event is not None:
                    new.append(event)
                none, zero = np.array([-1]), np.array([0.0])
                base = float(self.score(self.R[None], self.T[None], none, zero, logp, heat, roi)[0])
            elif k == 0 and abs(self.phi) < CANCEL:
                self.idle += 1
                if self.idle >= 3:  # 돌리려다 만 것으로 보인다. 그래도 놓친 회전이 없는지 확인한다
                    self.layer, self.phi, self.idle = -1, 0.0, 0
                    event = self._decide(logp, heat, roi, None)
                    if event is not None:
                        new.append(event)
            else:
                self.idle = 0
        if self.layer < 0:  # 방금 확정했더라도 다음 층이 이미 돌기 시작했을 수 있다
            self._probe_turn(base, logp, heat, roi)

        self.quality = self._quality(logp, heat, roi)
        self.doubt = self.doubt + 1 if self.layer < 0 and self.quality < DOUBT_QUALITY else 0
        if self.doubt >= DOUBT_FRAMES:
            event = self._decide(logp, heat, roi, None)
            if event is not None:
                new.append(event)
                self.doubt = 0
                self.quality = self._quality(logp, heat, roi)
        self.bad_frames = self.bad_frames + 1 if self.quality < LOST_QUALITY else 0
        if self.bad_frames >= LOST_FRAMES:  # 놓쳤다: 다음 프레임에 전체 화면에서 다시 찾는다
            self.R = self.T = None
            self.layer, self.phi, self.bad_frames = -1, 0.0, 0
        return new
