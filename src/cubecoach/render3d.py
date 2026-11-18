"""손에 든 큐브의 3차원 합성 렌더러.

연속 추적용 인식 모델의 학습 데이터를 만든다. 임의의 자세, 임의의 섞인 상태,
한 층을 돌리는 도중의 모습, 큐브를 쥔 손가락, 조명 변화를 그리고,
픽셀 단위 정답(어느 색 스티커인지 / 큐브 몸체인지 / 그 밖인지)과
스티커 중심 지도를 함께 낸다. 정답이 기하에서 바로 나오므로 라벨링 비용이 없다.

26개 조각을 각각 작은 정육면체로 그린다(먼 조각부터). 그래서 층을 돌리는
도중의 어긋난 모습과 조각끼리의 가림이 그대로 표현된다.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .cube import FACELET_INDEX, FACES, FRAMES, MOVES, SOLVED, to_ints
from .geometry import project, random_rotation, to_pixels, turn_matrix
from .synth import PALETTE, _to_linear, _to_srgb

N_CLASSES = 8  # 0~5: 스티커 색(WRGYOB), 6: 큐브 몸체, 7: 그 밖(배경·손)
BODY, OTHER = 6, 7
STRIDE = 4  # 정답 지도는 입력의 1/4 해상도
_FACE_MOVES = [m for m in MOVES if m[0] in FACES]

# 실제 촬영 영상(스티커리스 큐브, 흐린 실내광)에서 잰 칸 색의 중앙값 (sRGB 0~1).
# 빨강·주황은 색상 차가 15도 정도라 가장 헷갈리는 쌍이고, 노랑은 연두 쪽, 흰색은 푸른 쪽으로 찍힌다.
REAL_PALETTE = np.array(
    [(226, 233, 236), (214, 36, 52), (2, 168, 70), (200, 208, 44), (250, 104, 44), (4, 100, 188)], dtype=np.float32
) / 255.0

# 실제 영상에서 잘라 낸 '큐브가 없는' 조각 (scripts/extract_real_bg.py). 없으면 합성 배경만 쓴다
# (scripts/make_seg_data.py 는 이 조각이 없으면 멈춘다. 다른 실험이 되기 때문이다).
_REAL_BG_PATH = Path(__file__).resolve().parents[2] / "data" / "real" / "bg_patches.npy"
_real_bg: np.ndarray | None = None
_real_skin: np.ndarray | None = None  # 피부가 많이 보이는 조각의 번호


def real_bg() -> tuple[np.ndarray | None, np.ndarray | None]:
    global _real_bg, _real_skin
    if _real_bg is None and _REAL_BG_PATH.exists():
        _real_bg = np.load(_REAL_BG_PATH, mmap_mode="r")
        small = np.stack([cv2.resize(np.asarray(x), (32, 32), interpolation=cv2.INTER_AREA) for x in _real_bg])
        hsv = cv2.cvtColor(small.reshape(-1, 32, 3), cv2.COLOR_RGB2HSV).reshape(small.shape)
        skin = (hsv[..., 0] < 16) & (hsv[..., 1] > 50) & (hsv[..., 1] < 150) & (hsv[..., 2] > 90)
        _real_skin = np.nonzero(skin.mean((1, 2)) > 0.6)[0]
    return _real_bg, _real_skin


def _real_tex(idx: int, hi: int, rng: np.random.Generator) -> np.ndarray:
    """실제 조각 하나를 무작위로 자르고 뒤집어 (hi, hi) 선형 광량으로."""
    bg, _ = real_bg()
    x = np.asarray(bg[idx])
    s = int(rng.uniform(0.5, 1.0) * x.shape[0])
    a, b = rng.integers(0, x.shape[0] - s + 1, 2)
    x = x[a:a + s, b:b + s]
    if rng.random() < 0.5:
        x = x[:, ::-1]
    x = np.rot90(x, int(rng.integers(4)))
    x = cv2.resize(np.ascontiguousarray(x), (hi, hi), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    return _to_linear(x) * np.float32(rng.uniform(0.6, 1.4)) * rng.uniform(0.9, 1.1, 3).astype(np.float32)


def _cubie_faces():
    quads, normals, cubies, facelets = [], [], [], []
    axes = [np.array(v, dtype=np.float64) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    for p in itertools.product((-1, 0, 1), repeat=3):
        if not any(p):
            continue
        p = np.array(p, dtype=np.float64)
        for n in axes:
            t1 = np.array([n[1], n[2], n[0]])
            t2 = np.cross(n, t1)
            c = p + 0.5 * n
            quads.append([c + 0.5 * (a * t1 + b * t2) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            normals.append(n)
            cubies.append(p)
            key = (tuple(int(v) for v in p), tuple(int(v) for v in n))
            facelets.append(FACELET_INDEX.get(key, -1))
    return np.array(quads), np.array(normals), np.array(cubies), np.array(facelets)


QUADS, FACE_NORMALS, CUBIES, FACELET_OF = _cubie_faces()  # 26조각 x 6면 = 156면


@dataclass
class Background:
    base: np.ndarray
    grad: np.ndarray
    shapes: list  # (꼭짓점 4개의 화면 좌표, 색)
    blur: float
    real: int = -1  # 실제 배경 조각 번호 (-1이면 합성 배경)


@dataclass
class Look:
    """한 영상 안에서 변하지 않는 겉모습: 큐브 색감, 조명, 카메라 특성, 배경."""

    palette: np.ndarray  # (6, 3) 선형 광량
    body: float
    fill: float
    cut: float
    light: np.ndarray
    ambient: float
    gains: np.ndarray
    exposure: float
    wash: float
    blur: float
    noise: float
    jpeg: int
    glare: list  # (큐브 기준 위치 x, y, 크기 x, y, 세기)
    bg: Background
    logo: float = 0.0  # 흰 가운데 칸의 로고 크기 (0이면 없음)
    motion: float = 0.0  # 움직임 번짐 길이 (출력 해상도 픽셀)
    skin_tex: int = -1  # 손가락에 입힐 실제 피부 조각 번호


@dataclass
class Finger:
    theta: float  # 큐브 중심에서 본 방향
    r_out: float  # 바깥쪽 끝 (조각 단위)
    r_in: float  # 안쪽 끝
    dtheta: float
    width: float
    color: np.ndarray


@dataclass
class Scene:
    colors: np.ndarray  # (54,) 색 번호
    R: np.ndarray  # 큐브 → 카메라 회전
    T: np.ndarray  # 큐브 중심의 카메라 좌표
    f: float  # 초점거리 / 프레임 너비
    look: Look
    turn: tuple[str, float] | None = None  # (면, 각도) — 돌리는 중인 층
    fingers: list[Finger] = field(default_factory=list)

    @property
    def center_uv(self) -> np.ndarray:
        return self.f * self.T[:2] / self.T[2]

    @property
    def unit(self) -> float:
        """조각 한 변이 화면에서 차지하는 길이 (프레임 너비 대비)."""
        return self.f / self.T[2]


@dataclass
class Frame:
    img: np.ndarray  # (res, res, 3) uint8 RGB
    seg: np.ndarray  # (res/4, res/4) uint8 분류 정답
    heat: np.ndarray  # (res/4, res/4) float32 스티커 중심 지도 0~1
    centers: np.ndarray  # (54, 2) 정답 지도 좌표계의 스티커 중심
    vis: np.ndarray  # (54,) 스티커가 실제로 보이는 비율 0~1


# --- 무작위 장면 ----------------------------------------------------------

def sample_look(rng: np.random.Generator) -> Look:
    base = REAL_PALETTE if rng.random() < 0.6 else PALETTE
    hsv = cv2.cvtColor(base[None], cv2.COLOR_RGB2HSV)[0]
    hsv[:, 0] = (hsv[:, 0] + rng.normal(0, 3.0, 6)) % 360
    hsv[:, 1] = np.clip(hsv[:, 1] * rng.uniform(0.8, 1.05, 6), 0, 1)
    hsv[:, 2] = np.clip(hsv[:, 2] * rng.uniform(0.85, 1.0, 6), 0, 1)
    palette = _to_linear(cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB)[0])

    light = np.array([rng.normal(0, 0.5), rng.normal(-0.3, 0.5), 1.0])
    t, g = rng.normal(0, 0.16), rng.normal(0, 0.04)

    shapes = []
    for _ in range(rng.integers(5, 15)):
        c = rng.uniform(-0.6, 0.6, 2)
        w, h = rng.uniform(0.04, 0.6, 2)
        a = rng.uniform(0, np.pi)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        pts = np.array([(-w, -h), (w, -h), (w, h), (-w, h)]) / 2 @ rot.T + c
        kind = rng.random()
        if kind < 0.3:  # 큐브 색과 비슷한 방해물
            col = _to_linear(PALETTE[rng.integers(6)]) * rng.uniform(0.3, 1.0)
        elif kind < 0.5:  # 얼굴·팔 같은 살색
            col = _to_linear(np.array([0.85, 0.65, 0.52])) * rng.uniform(0.4, 1.1)
        else:
            col = _to_linear(rng.uniform(0, 1, 3)) * rng.uniform(0.3, 1.0)
        shapes.append((pts, col.astype(np.float32)))
    reals, skins = real_bg()
    bg = Background(
        base=(_to_linear(rng.uniform(0.05, 0.95, 3)) * rng.uniform(0.4, 1.0)).astype(np.float32),
        grad=rng.normal(0, 0.6, 2),
        shapes=shapes if reals is None or rng.random() < 0.5 else shapes[: rng.integers(0, 3)],
        blur=float(rng.uniform(0, 5)),
        real=int(rng.integers(len(reals))) if reals is not None and rng.random() < 0.55 else -1,
    )
    # 스티커리스 큐브: 칸이 면을 거의 다 덮고(틈만 보임) 모서리가 둥글다
    stickerless = rng.random() < 0.6
    glare = [
        (*rng.uniform(-1.8, 1.8, 2), *rng.uniform(0.4, 1.6, 2), rng.uniform(0.15, 0.8))
        for _ in range(rng.integers(0, 3))
    ]
    return Look(
        palette=palette.astype(np.float32),
        body=float(rng.uniform(0.01, 0.12) if stickerless else rng.uniform(0.004, 0.03)),
        fill=float(rng.uniform(0.94, 0.985) if stickerless else rng.uniform(0.78, 0.94)),
        cut=float(rng.uniform(0.10, 0.28) if stickerless else rng.uniform(0.04, 0.22)),
        light=light / np.linalg.norm(light),
        ambient=float(rng.uniform(0.35, 0.75)),
        gains=np.clip([1 + t, 1 + g, 1 - t], 0.5, 1.5).astype(np.float32),
        exposure=float(rng.uniform(0.5, 1.5)),
        wash=float(rng.uniform(0, 0.3)),
        blur=float(rng.uniform(0, 1.0)),
        noise=float(rng.uniform(0, 0.02)),
        jpeg=int(rng.uniform(45, 95)),
        glare=glare,
        bg=bg,
        logo=float(rng.uniform(0.35, 0.6)) if rng.random() < 0.5 else 0.0,
        motion=float(rng.uniform(2, 12)) if rng.random() < 0.35 else 0.0,
        skin_tex=int(rng.choice(skins)) if skins is not None and len(skins) and rng.random() < 0.5 else -1,
    )


def sample_fingers(rng: np.random.Generator) -> list[Finger]:
    """큐브를 쥔 손. 좌우와 아래에서 들어와 가장자리 스티커를 가린다."""
    out: list[Finger] = []
    if rng.random() < 0.12:
        return out
    tone = _to_linear(np.array([0.87, 0.67, 0.55])) * rng.uniform(0.45, 1.1)
    if rng.random() < 0.1:  # 장갑·소매 같은 살색이 아닌 가림
        tone = _to_linear(rng.uniform(0, 1, 3))
    sides = [0.0, np.pi, np.pi / 2]  # 오른쪽, 왼쪽, 아래 (화면은 y가 아래)
    for _ in range(rng.integers(1, 8)):
        out.append(Finger(
            theta=float(rng.choice(sides) + rng.normal(0, 0.55)),
            r_out=float(rng.uniform(3.2, 5.0)),
            r_in=float(rng.uniform(-1.2, 0.3) if rng.random() < 0.15 else rng.uniform(0.3, 2.0)),
            dtheta=float(rng.normal(0, 0.3)),
            width=float(rng.uniform(0.55, 1.0)),
            color=(tone * rng.uniform(0.85, 1.1)).astype(np.float32),
        ))
    if rng.random() < 0.65:  # 손바닥
        out.append(Finger(
            theta=float(rng.choice(sides) + rng.normal(0, 0.4)),
            r_out=float(rng.uniform(5.0, 7.0)),
            r_in=float(rng.uniform(1.4, 2.6)),
            dtheta=float(rng.normal(0, 0.15)),
            width=float(rng.uniform(2.0, 3.6)),
            color=(tone * rng.uniform(0.8, 1.05)).astype(np.float32),
        ))
    return out


def random_colors(rng: np.random.Generator) -> np.ndarray:
    """무작위로 섞인 상태. 다 맞췄거나 거의 맞춘 상태도 섞는다."""
    colors = to_ints(SOLVED)
    n = 0 if rng.random() < 0.06 else int(rng.integers(1, 6)) if rng.random() < 0.15 else 25
    for _ in range(n):
        colors = colors[MOVES[_FACE_MOVES[rng.integers(len(_FACE_MOVES))]]]
    return colors


def place(rng: np.random.Generator, f: float, face_frac: float, center: np.ndarray) -> np.ndarray:
    """화면에서 정면 한 면이 face_frac(프레임 너비 대비)만큼 보이도록 큐브를 놓는다."""
    tz = 3 * f / face_frac
    return np.array([center[0] * tz / f, center[1] * tz / f, tz])


def sample_scene(rng: np.random.Generator) -> tuple[Scene, tuple[float, float, float]]:
    """학습용 장면 하나와, 그 장면을 잘라 볼 관심영역."""
    f = float(rng.uniform(0.75, 1.15))
    face_frac = float(rng.uniform(0.10, 0.42))
    center = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2)])
    scene = Scene(
        colors=random_colors(rng),
        R=random_rotation(rng),
        T=place(rng, f, face_frac, center),
        f=f,
        look=sample_look(rng),
        fingers=sample_fingers(rng),
    )
    if rng.random() < 0.3:
        scene.turn = (FACES[rng.integers(6)], float(rng.uniform(0.08, np.pi / 2 - 0.08)))
    # 추적 중에는 큐브가 영역의 절반쯤을 채우고, 처음 찾을 때는 작게 보인다.
    size = face_frac * (rng.uniform(4.2, 9.0) if rng.random() < 0.15 else rng.uniform(2.0, 4.2))
    roi = (*(center + rng.normal(0, 0.08, 2) * size), float(size))
    return scene, roi


# --- 그리기 ---------------------------------------------------------------

def _poly(canvas: np.ndarray, pts: np.ndarray, color) -> None:
    cv2.fillConvexPoly(canvas, np.round(pts * 16).astype(np.int32), color, lineType=cv2.LINE_8, shift=4)


def _sticker_poly(q: np.ndarray, fill: float, cut: float) -> np.ndarray:
    """면 사각형을 안쪽으로 줄이고 모서리를 깎은 팔각형."""
    q = q.mean(0) + (q - q.mean(0)) * fill
    pts = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        pts += [a + cut * (b - a), b - cut * (b - a)]
    return np.array(pts)


def _area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def render(scene: Scene, roi: tuple[float, float, float], rng: np.random.Generator,
           res: int = 192, ss: int = 2, version: int = 2) -> Frame:
    """version 3: 층을 돌리는 장면에서 화면 전체 대신 돌아가는 층만 번지게 한다 (실제 영상에서는 그 층만 번진다).
    version 2 의 난수 사용 순서는 그대로라 v2 데이터는 같은 시드로 바이트까지 다시 만들어진다."""
    hi, look = res * ss, scene.look
    n_lab = res // STRIDE
    k = hi // n_lab  # 정답 지도 한 칸에 해당하는 고해상도 픽셀 수

    quads, normals, cubies = QUADS, FACE_NORMALS, CUBIES
    layer = None
    if scene.turn is not None:
        face, ang = scene.turn
        layer = CUBIES @ np.array(FRAMES[face][0], dtype=np.float64) > 0.5
        M = turn_matrix(face, ang)
        quads, normals, cubies = quads.copy(), normals.copy(), cubies.copy()
        quads[layer] = quads[layer] @ M.T
        normals[layer] = normals[layer] @ M.T
        cubies[layer] = cubies[layer] @ M.T

    Xc = quads @ scene.R.T + scene.T  # (156, 4, 3)
    nc = normals @ scene.R.T
    fc = Xc.mean(1)
    facing = -(nc * fc).sum(1) / np.linalg.norm(fc, axis=1)
    depth = np.linalg.norm(cubies @ scene.R.T + scene.T, axis=1)
    px = to_pixels(project(Xc, scene.f), roi, hi)  # (156, 4, 2)

    # 배경
    img = np.empty((hi, hi, 3), dtype=np.float32)
    ys, xs = np.mgrid[0:hi, 0:hi].astype(np.float32)
    u = (xs / hi - 0.5) * roi[2] + roi[0]
    v = (ys / hi - 0.5) * roi[2] + roi[1]
    if look.bg.real >= 0:
        img[:] = _real_tex(look.bg.real, hi, rng)
    else:
        img[:] = look.bg.base * np.clip(1 + look.bg.grad[0] * u + look.bg.grad[1] * v, 0.3, 1.8)[..., None]
    for pts, col in look.bg.shapes:
        _poly(img, to_pixels(pts, roi, hi), tuple(float(c) for c in col))
    if look.bg.blur > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), look.bg.blur * hi / 384 / max(roi[2], 0.2))
    seg = np.full((hi, hi), OTHER, dtype=np.uint8)
    ids = np.zeros((hi, hi), dtype=np.uint8)
    lmask = np.zeros((hi, hi), dtype=np.float32) if (version >= 3 and layer is not None) else None  # 돌아가는 층이 보이는 곳

    # 큐브: 먼 조각부터, 카메라를 향한 면만
    centers = np.zeros((54, 2), dtype=np.float32)
    areas = np.zeros(54, dtype=np.float32)
    for i in np.argsort(-depth, kind="stable"):
        if facing[i] <= 0:
            continue
        shade = look.ambient + (1 - look.ambient) * max(0.0, float(-nc[i] @ look.light))
        _poly(img, px[i], (look.body * shade,) * 3)
        _poly(seg, px[i], BODY)
        _poly(ids, px[i], 0)
        if lmask is not None:
            _poly(lmask, px[i], 1.0 if layer[i] else 0.0)
        j = FACELET_OF[i]
        if j < 0:
            continue
        pts = _sticker_poly(px[i], look.fill, look.cut)
        col = look.palette[scene.colors[j]] * shade
        _poly(img, pts, tuple(float(c) for c in col))
        _poly(seg, pts, int(scene.colors[j]))
        _poly(ids, pts, int(j) + 1)
        if look.logo and j % 9 == 4 and scene.colors[j] == 0:
            c0 = px[i].mean(0)
            disc = c0 + (px[i] - c0) * look.logo
            ring = _sticker_poly(disc, 1.0, 0.3)
            _poly(img, ring, tuple(float(c) * 0.8 for c in col))
            mark = c0 + (px[i] - c0) * look.logo * 0.55
            _poly(img, _sticker_poly(mark, 1.0, 0.45), tuple(float(c) for c in _to_linear(np.array([0.25, 0.45, 0.85])) * shade))
        centers[j] = px[i].mean(0) / k
        areas[j] = _area(pts)

    # 반사광 (큐브 위에만)
    cu, cv_ = scene.center_uv
    for gx, gy, sx, sy, a in look.glare:
        bu, bv = cu + gx * scene.unit, cv_ + gy * scene.unit
        blob = np.exp(-(((u - bu) / (sx * scene.unit)) ** 2 + ((v - bv) / (sy * scene.unit)) ** 2))
        alpha = (blob * a * (seg != OTHER))[..., None]
        img = img * (1 - alpha) + alpha

    # 손: 그림자를 먼저 드리우고 그 위에 손가락을 그린다
    if scene.fingers:
        shadow = np.zeros((hi, hi), dtype=np.float32)
        lines = []
        for fg in scene.fingers:
            d0 = np.array([np.cos(fg.theta), np.sin(fg.theta)])
            d1 = np.array([np.cos(fg.theta + fg.dtheta), np.sin(fg.theta + fg.dtheta)])
            p0 = to_pixels(scene.center_uv + scene.unit * fg.r_out * d0, roi, hi)
            p1 = to_pixels(scene.center_uv + scene.unit * fg.r_in * d1, roi, hi)
            w = max(1, int(scene.unit * fg.width / roi[2] * hi))
            a, b = tuple(int(c) for c in np.clip(p0, -4 * hi, 5 * hi)), tuple(int(c) for c in np.clip(p1, -4 * hi, 5 * hi))
            lines.append((a, b, w, fg.color))
            cv2.line(shadow, a, b, 1.0, int(w * 1.5))
        shadow = cv2.GaussianBlur(shadow, (0, 0), max(1.0, hi / 60))
        img *= (1 - 0.45 * shadow)[..., None]
        if look.skin_tex >= 0:
            mask = np.zeros((hi, hi), dtype=np.float32)
            for a, b, w, col in lines:
                cv2.line(mask, a, b, 1.0, w)
                cv2.line(seg, a, b, OTHER, w)
                cv2.line(ids, a, b, 0, w)
            edge = cv2.GaussianBlur(mask, (0, 0), max(1.0, hi / 120))  # 가장자리는 조금 어둡게
            tex = _real_tex(look.skin_tex, hi, rng) * (0.6 + 0.4 * edge)[..., None]
            img = img * (1 - mask[..., None]) + tex * mask[..., None]
        else:
            for a, b, w, col in lines:
                cv2.line(img, a, b, tuple(float(c) * 0.8 for c in col), w)
                cv2.line(img, a, b, tuple(float(c) for c in col), max(1, int(w * 0.6)))
                cv2.line(seg, a, b, OTHER, w)
                cv2.line(ids, a, b, 0, w)

    # 카메라
    out = _to_srgb(img * look.gains * look.exposure)
    gray = out.mean(axis=2, keepdims=True)
    out = out * (1 - look.wash) + gray * look.wash
    if version >= 3:
        blur = look.blur
    else:
        blur = look.blur + (0.8 if scene.turn is not None else 0.0) * rng.random()  # 돌리는 중엔 조금 더 흐리다
    if blur > 0.2:
        out = cv2.GaussianBlur(out, (0, 0), blur * ss)
    if lmask is not None and rng.random() < 0.6:
        # 돌아가는 층만 회전 방향으로 번진다: 층의 회전축을 화면에 투영한 방향에 수직으로
        face, _ = scene.turn
        ax = np.array(FRAMES[face][0], dtype=np.float64)
        c0 = (ax * 1.0) @ scene.R.T + scene.T
        c1 = (ax * 1.6) @ scene.R.T + scene.T
        p0, p1 = to_pixels(project(c0[None], scene.f), roi, hi)[0], to_pixels(project(c1[None], scene.f), roi, hi)[0]
        dvec = p1 - p0
        ang_deg = float(np.degrees(np.arctan2(dvec[1], dvec[0]))) + 90.0 if np.hypot(*dvec) > 1e-3 else float(rng.uniform(0, 180))
        L = int(rng.uniform(3, 14) * ss) | 1
        ker = np.zeros((L, L), dtype=np.float32)
        ker[L // 2] = 1.0
        ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), ang_deg, 1.0), (L, L))
        ker /= max(ker.sum(), 1e-6)
        mb = cv2.filter2D(lmask, -1, ker)[..., None]
        out = out * (1 - mb) + cv2.filter2D(out, -1, ker) * mb
    if version >= 3:
        motion = look.motion * rng.uniform(0.5, 1.0) if look.motion else 0.0  # 큐브를 옮기거나 흔들 때만 화면 전체
    else:
        motion = look.motion * rng.uniform(0.5, 1.0) if look.motion else (rng.uniform(2, 10) if scene.turn is not None and rng.random() < 0.4 else 0.0)
    if motion >= 1.5:  # 빠르게 돌리거나 흔들 때의 한 방향 번짐
        L = int(motion * ss) | 1
        ker = np.zeros((L, L), dtype=np.float32)
        ker[L // 2] = 1.0
        ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), float(rng.uniform(0, 180)), 1.0), (L, L))
        out = cv2.filter2D(out, -1, ker / max(ker.sum(), 1e-6))
    out = cv2.resize(out, (res, res), interpolation=cv2.INTER_AREA)
    out = out + rng.normal(0, look.noise, out.shape)
    u8 = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, look.jpeg])
    u8 = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # 정답 지도: 칸마다 가장 넓게 차지한 분류
    counts = np.stack([(seg == c).reshape(n_lab, k, n_lab, k).sum((1, 3)) for c in range(N_CLASSES)])
    seg_lab = counts.argmax(0).astype(np.uint8)

    # 스티커가 실제로 보이는 비율과 중심 지도
    seen = np.bincount(ids.ravel(), minlength=55)[1:].astype(np.float32)
    vis = np.where(areas > 0, np.clip(seen / np.maximum(areas, 1e-6), 0, 1), 0).astype(np.float32)
    heat = np.zeros((n_lab, n_lab), dtype=np.float32)
    gy, gx = np.mgrid[0:n_lab, 0:n_lab].astype(np.float32) + 0.5
    for j in np.nonzero(vis > 0.5)[0]:
        cx, cy = centers[j]
        if not (0 <= cx < n_lab and 0 <= cy < n_lab):
            continue
        if ids[min(int(cy * k), hi - 1), min(int(cx * k), hi - 1)] != j + 1:
            continue  # 중심이 가려진 스티커는 중심 지도에 넣지 않는다
        sig = float(np.clip(0.2 * np.sqrt(areas[j]) / k, 0.5, 3.0))
        heat = np.maximum(heat, np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sig * sig)))
    return Frame(u8, seg_lab, heat, centers, vis)
