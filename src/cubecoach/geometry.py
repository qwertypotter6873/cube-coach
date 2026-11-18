"""큐브의 3차원 기하와 카메라 투영.

단위: 조각 하나의 변 = 1, 큐브 중심 = 원점 (큐브 한 변 = 3).
큐브 좌표축은 cube.py와 같다 (x=오른쪽 R, y=위 U, z=앞 F).
카메라 좌표축은 x=오른쪽, y=아래, z=카메라가 보는 방향.
화면 좌표는 프레임 너비로 나눈 값이고 프레임 중심이 (0, 0)이다.
"""

from __future__ import annotations

import numpy as np

from .cube import FACELETS, FACES, FRAMES

# 54칸 각각의 스티커 중심·법선·면 안에서의 오른쪽/아래 방향 (큐브 좌표)
NORMALS = np.array([n for _, n in FACELETS], dtype=np.float64)
CENTERS = np.array([p for p, _ in FACELETS], dtype=np.float64) + 0.5 * NORMALS
RIGHTS = np.array([FRAMES[FACES[i // 9]][1] for i in range(54)], dtype=np.float64)
DOWNS = np.array([FRAMES[FACES[i // 9]][2] for i in range(54)], dtype=np.float64)

# 큐브 앞면(F)을 정면에서, 윗면(U)이 위로 오게 보는 자세
FRONT_VIEW = np.diag([1.0, -1.0, -1.0])


def rotvec(w: np.ndarray) -> np.ndarray:
    """회전 벡터(축 × 각도) → 회전 행렬. (..., 3) → (..., 3, 3)."""
    w = np.asarray(w, dtype=np.float64)
    theta = np.linalg.norm(w, axis=-1)[..., None, None]
    k = w / np.maximum(theta[..., 0], 1e-12)
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def random_rotation(rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    """균등 분포의 무작위 회전 행렬. n을 주면 (n, 3, 3)."""
    q = rng.normal(size=(n or 1, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)
    return R if n else R[0]


def rotation_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """두 자세 사이의 각도(도)."""
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def turn_matrix(face: str, angle: float) -> np.ndarray:
    """face 층을 바깥에서 봤을 때 시계 방향으로 angle(rad)만큼 돌리는 회전 (큐브 좌표)."""
    return rotvec(-angle * np.array(FRAMES[face][0], dtype=np.float64))


def project(X: np.ndarray, f: float) -> np.ndarray:
    """카메라 좌표 (..., 3) → 화면 좌표 (..., 2)."""
    return f * X[..., :2] / X[..., 2:3]


def to_pixels(uv: np.ndarray, roi: tuple[float, float, float], n: int) -> np.ndarray:
    """화면 좌표 → 정사각 관심영역(roi = 중심 x, 중심 y, 한 변) 안의 픽셀 좌표 (n x n)."""
    cx, cy, size = roi
    return ((uv - np.array([cx, cy])) / size + 0.5) * n
