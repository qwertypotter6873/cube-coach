"""손에 든 큐브를 돌리는 합성 영상. 추적기를 정답과 대조해 시험하는 데 쓴다.

실제 손놀림에서 추적을 어렵게 하는 요소를 넣는다.
- 큐브가 계속 조금씩 흔들리고 움직인다.
- 손가락이 스티커를 가리고, 가끔 고쳐 쥔다.
- 회전은 여러 프레임에 걸쳐 일어난다 (도중에는 층이 어긋나 보인다).
- 한 층만 돌리기도 하고, 그 층을 잡은 채 나머지 두 층을 돌리기도 한다.
- 가끔 큐브를 통째로 돌려 다른 면을 본다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator

import numpy as np

from .cube import FACES, MOVES
from .geometry import random_rotation, rotvec, turn_matrix
from .render3d import Scene, _FACE_MOVES, place, sample_fingers, sample_look

_NORMALS = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.float64)
_K = {"": 1, "2": 2, "'": 3}
_ANGLE = {1: np.pi / 2, 2: np.pi, 3: -np.pi / 2}


@dataclass
class Step:
    scene: Scene
    done: list[tuple[int, int]]  # 이 프레임에 끝난 회전: (가운데 색 번호, 시계 방향 90도 단위)


def faces_in_view(R: np.ndarray, min_facing: float = 0.3) -> int:
    """카메라 쪽을 충분히 향한 면의 수 (큐브가 화면 가운데 근처에 있다고 보고 어림한다)."""
    return int((-(R @ _NORMALS.T)[2] > min_facing).sum())


def simulate(rng: np.random.Generator, n_moves: int = 20, fingers: bool = True, any_view: bool = False,
             turn_frames: tuple[int, int] = (5, 11), pause_frames: tuple[int, int] = (3, 10),
             regrip: bool = True) -> Iterator[Step]:
    """any_view=False(기본)면 두 면 이상이 보이도록 든다 — 앱이 사용자에게 안내할 자세.
    True면 한 면만 정면으로 보이는 경우까지 포함한다 (보이지 않는 층의 회전이 생긴다).

    turn_frames: 90도 회전 하나에 걸리는 프레임 수의 범위 [최소, 최대).
    pause_frames: 회전과 회전 사이에 멈춰 있는 프레임 수의 범위 [최소, 최대).
    둘을 줄이면 빠른 손놀림(또는 낮은 프레임률)이 된다. 회전이 끝난 모습은 항상 1프레임은 나온다."""
    look = sample_look(rng)
    f = float(rng.uniform(0.8, 1.1))
    face_frac = float(rng.uniform(0.2, 0.34))
    center = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.1, 0.1)])
    R = random_rotation(rng)
    while not any_view and faces_in_view(R, 0.4) < 2:
        R = random_rotation(rng)
    colors = np.arange(54) // 9
    for _ in range(25):
        colors = colors[MOVES[_FACE_MOVES[rng.integers(18)]]]
    hand = sample_fingers(rng) if fingers else []
    omega, vel = np.zeros(3), np.zeros(2)

    def drift() -> None:
        nonlocal R, omega, vel, center
        omega = 0.85 * omega + rng.normal(0, 0.012, 3)  # 프레임당 1~2도쯤 흔들린다
        if not any_view and faces_in_view(rotvec(omega) @ R) < 2 <= faces_in_view(R):
            omega = -0.5 * omega  # 한 면만 보이는 쪽으로는 흔들리지 않는다
        R = rotvec(omega) @ R
        vel = 0.85 * vel + rng.normal(0, 0.0012, 2)
        center = np.clip(center + vel, [-0.25, -0.15], [0.25, 0.15])
        for fg in hand:
            fg.theta += rng.normal(0, 0.012)
            fg.r_in = float(np.clip(fg.r_in + rng.normal(0, 0.025), 0.3, 2.6))

    def scene(R_now: np.ndarray, turn=None) -> Scene:
        return Scene(colors=colors.copy(), R=R_now.copy(), T=place(rng, f, face_frac, center), f=f,
                     look=look, turn=turn, fingers=[replace(fg) for fg in hand])

    def hold(n: int) -> Iterator[Step]:
        for _ in range(n):
            drift()
            yield Step(scene(R), [])

    yield from hold(12)
    last = ""
    for _ in range(n_moves):
        m = _FACE_MOVES[rng.integers(18)]
        while m[0] == last:
            m = _FACE_MOVES[rng.integers(18)]
        last = m[0]
        yield from hold(int(rng.integers(*pause_frames)))

        if regrip and rng.random() < 0.15:  # 큐브를 통째로 90도쯤 돌려 다른 면을 본다
            axis = R @ np.eye(3)[rng.integers(3)] * rng.choice([-1, 1])
            n = int(rng.integers(10, 17))
            for _ in range(n):
                drift()
                R = rotvec(axis * (np.pi / 2 / n)) @ R
                yield Step(scene(R), [])
        if fingers and regrip and rng.random() < 0.12:  # 고쳐 쥐기
            hand = sample_fingers(rng)

        face, k = m[0], _K[m[1:]]
        n = int(rng.integers(*turn_frames) * (1.6 if k == 2 else 1))
        layer_held = rng.random() < 0.3  # 돌리는 층을 잡고 나머지를 돌리는 경우
        for t in range(1, n + 1):
            x = t / (n + 1)
            phi = _ANGLE[k] * x * x * (3 - 2 * x)
            drift()
            R_now = R @ turn_matrix(face, phi).T if layer_held else R
            yield Step(scene(R_now, (face, float(phi))), [])
        center_color = int(colors[FACES.index(face) * 9 + 4])
        colors = colors[MOVES[m]]
        if layer_held:
            R = R @ turn_matrix(face, _ANGLE[k]).T
        drift()
        yield Step(scene(R), [(center_color, k)])
    yield from hold(12)
