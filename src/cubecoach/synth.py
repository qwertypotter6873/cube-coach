"""큐브 한 면의 합성 이미지 생성기.

직접 촬영한 데이터가 모이기 전에 파이프라인을 검증하고, 이후에는 사전학습용으로 쓴다.
실제 카메라에서 색 인식을 망가뜨리는 요인을 무작위로 섞는다:
조명 색온도(전구색 아래에서 흰색↔노랑, 빨강↔주황이 가까워진다), 그림자, 반사광,
가이드 박스에 대충 맞춘 정렬 오차, 손가락 가림, 흐림·노이즈·JPEG 압축.

라벨은 이미지를 정면으로 봤을 때 왼쪽 위부터 행 우선 9칸, 값은 cube.COLORS의 색 번호.
"""

from __future__ import annotations

import cv2
import numpy as np

from .cube import COLORS

# 표준 배색의 sRGB 기준값 (COLORS = "WRGYOB" 순서)
PALETTE = np.array(
    [
        (242, 242, 238),  # W
        (196, 30, 36),  # R
        (0, 155, 72),  # G
        (255, 213, 0),  # Y
        (255, 110, 10),  # O
        (0, 70, 173),  # B
    ],
    dtype=np.float32,
) / 255.0

FACE_FILL = 0.89  # 가이드에 맞췄을 때 면이 크롭에서 차지하는 평균 비율
_CANVAS = 160


def cell_centers(size: int) -> np.ndarray:
    """정렬이 완벽하다고 가정했을 때 9칸 중심의 (x, y). 고전 베이스라인이 이 위치만 본다."""
    m = (1 - FACE_FILL) / 2
    c = (m + (np.arange(3) + 0.5) * FACE_FILL / 3) * size
    return np.array([(c[col], c[row]) for row in range(3) for col in range(3)], dtype=np.float32)


def random_labels(rng: np.random.Generator) -> np.ndarray:
    """칸마다 독립인 무작위 색. 거의 맞춘 큐브처럼 한두 색뿐인 면도 섞는다."""
    r = rng.random()
    if r < 0.12:
        return np.full(9, rng.integers(6))
    if r < 0.30:
        return rng.choice(rng.choice(6, size=2, replace=False), size=9)
    return rng.integers(0, 6, size=9)


def _to_linear(x: np.ndarray) -> np.ndarray:
    return np.power(np.clip(x, 0, 1), 2.2)


def _to_srgb(x: np.ndarray) -> np.ndarray:
    return np.power(np.clip(x, 0, 1), 1 / 2.2)


def _cube_palette(rng: np.random.Generator) -> np.ndarray:
    """큐브 제품마다 다른 색감: 색상·채도·명도를 조금씩 흔든다."""
    hsv = cv2.cvtColor(PALETTE[None], cv2.COLOR_RGB2HSV)[0]  # H: 0~360
    hsv[:, 0] = (hsv[:, 0] + rng.normal(0, 3.0, 6)) % 360
    hsv[:, 1] = np.clip(hsv[:, 1] * rng.uniform(0.8, 1.05, 6), 0, 1)
    hsv[:, 2] = np.clip(hsv[:, 2] * rng.uniform(0.85, 1.0, 6), 0, 1)
    return cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB)[0]


def _draw_face(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    s = _CANVAS
    pal = _cube_palette(rng)
    img = np.full((s, s, 3), rng.uniform(0.02, 0.14), dtype=np.float32)  # 검은 플라스틱 몸체
    cell = s / 3
    gap = cell * rng.uniform(0.03, 0.12)
    rad = int(cell * rng.uniform(0.04, 0.28))
    for k, lab in enumerate(labels):
        row, col = divmod(k, 3)
        x0, y0 = int(col * cell + gap), int(row * cell + gap)
        x1, y1 = int((col + 1) * cell - gap), int((row + 1) * cell - gap)
        color = tuple(float(v) for v in pal[lab])
        cv2.rectangle(img, (x0 + rad, y0), (x1 - rad, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + rad), (x1, y1 - rad), color, -1)
        for cx, cy in ((x0 + rad, y0 + rad), (x1 - rad, y0 + rad), (x0 + rad, y1 - rad), (x1 - rad, y1 - rad)):
            cv2.circle(img, (cx, cy), rad, color, -1, cv2.LINE_AA)
    if rng.random() < 0.3:  # 가운데 칸의 로고
        c = int(s / 2)
        r = int(cell * rng.uniform(0.12, 0.3))
        cv2.circle(img, (c, c), r, tuple(float(v) for v in rng.uniform(0, 1, 3)), -1, cv2.LINE_AA)
    return img


def _light(img: np.ndarray, rng: np.random.Generator, d: float) -> np.ndarray:
    """선형 광량 공간에서 조명을 입힌다."""
    s = img.shape[0]
    lin = _to_linear(img)
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) / s - 0.5

    # 한쪽이 어두운 조명 기울기
    ang = rng.uniform(0, 2 * np.pi)
    grad = 1 + rng.uniform(0, 0.6) * d * (np.cos(ang) * xx + np.sin(ang) * yy)
    lin *= np.clip(grad, 0.25, 1.6)[..., None]

    # 손·팔이 드리운 그림자 (경계가 부드러운 반평면)
    if rng.random() < 0.4:
        ang = rng.uniform(0, 2 * np.pi)
        dist = np.cos(ang) * xx + np.sin(ang) * yy - rng.uniform(-0.3, 0.3)
        mask = 1 / (1 + np.exp(-dist / rng.uniform(0.02, 0.12)))
        lin *= (1 - mask * rng.uniform(0.25, 0.65) * min(d, 1.3))[..., None]

    # 광택 플라스틱의 반사광
    for _ in range(rng.integers(0, 3)):
        cx, cy = rng.uniform(-0.5, 0.5, 2)
        sx, sy = rng.uniform(0.05, 0.25, 2)
        blob = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        a = (blob * rng.uniform(0.2, 0.85) * min(d, 1.2))[..., None]
        lin = lin * (1 - a) + a

    # 조명 색온도와 색조 (전구색: 빨강↑ 파랑↓)
    t = rng.normal(0, 0.2 * d)
    g = rng.normal(0, 0.05 * d)
    gains = np.array([1 + t, 1 + g, 1 - t], dtype=np.float32)
    lin *= np.clip(gains, 0.45, 1.6)

    lin *= rng.uniform(0.45, 1.5)  # 노출
    out = _to_srgb(lin)
    wash = rng.uniform(0, 0.35) * min(d, 1.3)  # 웹캠의 물 빠진 색
    gray = out.mean(axis=2, keepdims=True)
    return out * (1 - wash) + gray * wash


def _background(size: int, rng: np.random.Generator) -> np.ndarray:
    kind = rng.integers(4)
    base = rng.uniform(0, 1, 3).astype(np.float32)
    if kind == 0:  # 살색 (손)
        base = np.array([0.85, 0.65, 0.52], dtype=np.float32) * rng.uniform(0.6, 1.1)
    bg = np.tile(base, (size, size, 1))
    if kind == 1:  # 옆면이 살짝 보이는 경우: 큐브 색 타일
        t = size // 3
        for r in range(3):
            for c in range(3):
                bg[r * t : (r + 1) * t, c * t : (c + 1) * t] = PALETTE[rng.integers(6)] * rng.uniform(0.4, 1.0)
    if kind == 2:  # 그라데이션 벽
        ramp = np.linspace(0.5, 1.2, size, dtype=np.float32)
        bg = bg * (ramp[:, None, None] if rng.random() < 0.5 else ramp[None, :, None])
    bg += rng.normal(0, 0.03, bg.shape).astype(np.float32)
    return np.clip(bg, 0, 1)


def render_face(
    labels: np.ndarray, rng: np.random.Generator, size: int = 96, difficulty: float = 1.0
) -> np.ndarray:
    """라벨 9개 → (size, size, 3) uint8 RGB. difficulty가 클수록 조명·정렬이 나쁘다."""
    d = difficulty
    face = _light(_draw_face(labels, rng), rng, d)

    # 가이드 박스에 손으로 맞춘 정렬: 크기·위치·회전·기울기가 조금씩 어긋난다.
    scale = rng.uniform(0.78, 1.0)
    half = scale * size / 2
    ang = np.deg2rad(rng.normal(0, 5 * d))
    center = size / 2 + rng.normal(0, 0.03 * d * size, 2)
    corners = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=np.float32) * half
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=np.float32)
    dst = corners @ rot.T + center + rng.normal(0, 0.02 * d * size, (4, 2))
    c = _CANVAS
    src = np.array([(0, 0), (c, 0), (c, c), (0, c)], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))

    # 면과 불투명도 마스크를 따로 변환해 배경 위에 얹는다 (경계가 자연스럽게 섞인다).
    warped = cv2.warpPerspective(face.astype(np.float32), H, (size, size), flags=cv2.INTER_LINEAR)
    alpha = cv2.warpPerspective(np.ones((c, c), np.float32), H, (size, size), flags=cv2.INTER_LINEAR)[..., None]
    out = np.ascontiguousarray(_background(size, rng) * (1 - alpha) + warped, dtype=np.float32)

    # 큐브를 쥔 손가락이 가장자리를 가린다.
    for _ in range(rng.integers(0, 3) if rng.random() < 0.35 else 0):
        side = rng.integers(4)
        along = rng.uniform(0.1, 0.9) * size
        cx, cy = [(along, 0), (size, along), (along, size), (0, along)][side]
        axes = (int(rng.uniform(0.05, 0.11) * size), int(rng.uniform(0.08, 0.17) * size))
        skin = np.array([0.85, 0.65, 0.52]) * rng.uniform(0.55, 1.1)
        cv2.ellipse(out, (int(cx), int(cy)), axes, 90.0 * (side % 2), 0, 360, tuple(float(v) for v in skin), -1, cv2.LINE_AA)

    # 카메라: 초점·움직임 흐림, 센서 노이즈, JPEG 압축
    sigma = rng.uniform(0, 1.1) * d
    if sigma > 0.2:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    out = out + rng.normal(0, rng.uniform(0, 0.025) * d, out.shape)
    u8 = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(40, 95))])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def make_dataset(n: int, seed: int, size: int = 96, difficulty: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(n, size, size, 3) uint8 이미지와 (n, 9) 라벨."""
    rng = np.random.default_rng(seed)
    xs = np.empty((n, size, size, 3), dtype=np.uint8)
    ys = np.empty((n, 9), dtype=np.int64)
    for i in range(n):
        ys[i] = random_labels(rng)
        xs[i] = render_face(ys[i], rng, size, difficulty)
    return xs, ys


assert len(COLORS) == len(PALETTE)
