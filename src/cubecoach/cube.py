"""3x3x3 큐브 상태 모델.

카메라는 한 번에 최대 세 면만 본다. 나머지는 이 모델이 계산한다:
여섯 면을 한 번 스캔해 54칸 전체를 알고 나면, 이후의 모든 회전은
54칸의 순열이므로 보이지 않는 면도 항상 정확히 알 수 있다.

표기: 면 순서는 U R F D L B, 각 면은 그 면을 정면으로 봤을 때 왼쪽 위부터
행 우선으로 9칸. 색은 표준 배색(흰색 위, 초록 앞)을 쓴다.
회전 순열은 손으로 적지 않고 3차원 좌표에서 계산한다.
"""

from __future__ import annotations

import random

import numpy as np

FACES = "URFDLB"
COLORS = "WRGYOB"  # U=흰 R=빨 F=초 D=노 L=주 B=파
COLOR_NAMES_KO = {"W": "흰색", "R": "빨강", "G": "초록", "Y": "노랑", "O": "주황", "B": "파랑"}
SOLVED = "".join(c * 9 for c in COLORS)

# 면마다 (법선, 오른쪽 방향, 아래 방향). 좌표축: x=오른쪽, y=위, z=앞.
_FRAMES = {
    "U": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    "R": ((1, 0, 0), (0, 0, -1), (0, -1, 0)),
    "F": ((0, 0, 1), (1, 0, 0), (0, -1, 0)),
    "D": ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    "L": ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "B": ((0, 0, -1), (-1, 0, 0), (0, -1, 0)),
}


def _facelets() -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """54칸 각각의 (조각 위치, 스티커 법선)."""
    out = []
    for f in FACES:
        n, right, down = (np.array(v) for v in _FRAMES[f])
        for row in range(3):
            for col in range(3):
                pos = n + (col - 1) * right + (row - 1) * down
                out.append((tuple(int(v) for v in pos), tuple(int(v) for v in n)))
    return out


_FACELETS = _facelets()
_INDEX = {fl: i for i, fl in enumerate(_FACELETS)}
FACELETS, FACELET_INDEX, FRAMES = _FACELETS, _INDEX, _FRAMES  # 3차원 기하 모듈이 쓴다


def _rot90(axis: np.ndarray, v: np.ndarray) -> np.ndarray:
    """v를 axis 둘레로, 축 바깥에서 봤을 때 시계 방향으로 90도 돌린다."""
    return np.cross(v, axis) + axis * int(np.dot(axis, v))


def _turn_perm(axis: tuple[int, int, int], layers: tuple[int, ...]) -> np.ndarray:
    """new[i] = old[perm[i]] 꼴의 순열. layers는 축 방향 좌표(-1, 0, 1)의 부분집합."""
    a = np.array(axis)
    perm = np.arange(54)
    for i, (pos, nrm) in enumerate(_FACELETS):
        if int(np.dot(a, pos)) in layers:
            new_pos = tuple(int(v) for v in _rot90(a, np.array(pos)))
            new_nrm = tuple(int(v) for v in _rot90(a, np.array(nrm)))
            perm[_INDEX[(new_pos, new_nrm)]] = i
    return perm


def _build_moves() -> dict[str, np.ndarray]:
    base = {f: _turn_perm(_FRAMES[f][0], (1,)) for f in FACES}
    # 큐브 통째 회전: x는 R 방향, y는 U 방향, z는 F 방향.
    base["x"] = _turn_perm(_FRAMES["R"][0], (-1, 0, 1))
    base["y"] = _turn_perm(_FRAMES["U"][0], (-1, 0, 1))
    base["z"] = _turn_perm(_FRAMES["F"][0], (-1, 0, 1))
    moves = {}
    for name, p in base.items():
        moves[name] = p
        moves[name + "2"] = p[p]
        moves[name + "'"] = p[p][p]
    return moves


MOVES = _build_moves()
FACE_MOVES = [m for m in MOVES if m[0] in FACES]


def apply(state: str, alg: str | list[str]) -> str:
    """회전 기호열(예: "R U R' U'")을 상태에 적용한다."""
    seq = alg.split() if isinstance(alg, str) else alg
    arr = np.array(list(state))
    for m in seq:
        arr = arr[MOVES[m]]
    return "".join(arr)


def to_ints(state: str) -> np.ndarray:
    """상태 문자열 → 색 번호 54개. 순열(MOVES)을 그대로 인덱싱에 쓸 수 있다."""
    return np.array([COLORS.index(c) for c in state], dtype=np.int64)


def to_str(colors: np.ndarray) -> str:
    return "".join(COLORS[int(c)] for c in colors)


def scramble(n: int = 25, rng: random.Random | None = None) -> list[str]:
    """같은 면이 연달아 나오지 않는 무작위 섞기."""
    rng = rng or random.Random()
    seq: list[str] = []
    while len(seq) < n:
        m = rng.choice(FACE_MOVES)
        if seq and seq[-1][0] == m[0]:
            continue
        seq.append(m)
    return seq


def face(state: str, f: str) -> str:
    """한 면의 9칸."""
    i = FACES.index(f) * 9
    return state[i : i + 9]


def rotate_face_cw(nine: str) -> str:
    """3x3 격자를 시계 방향으로 90도 돌린 모습."""
    return "".join(nine[i] for i in (6, 3, 0, 7, 4, 1, 8, 5, 2))


# --- 유효성 검사 ---------------------------------------------------------
# 스캔 결과가 "실제로 존재할 수 있는 큐브"인지 확인한다. 한 칸만 잘못 읽어도
# 여기서 걸리므로, 인식 오류를 잡아내는 안전망이자 데이터 라벨 검증기다.

def _clockwise(piece: tuple[int, ...]) -> tuple[int, ...]:
    """모서리 조각의 세 칸을 바깥에서 봤을 때 시계 방향 순서로 정렬한다."""
    n = [np.array(_FACELETS[i][1]) for i in piece]
    pos = np.array(_FACELETS[piece[0]][0])
    if np.dot(np.cross(n[0], n[1]), pos) > 0:  # 0→1→2가 반시계면 뒤집는다
        return (piece[0], piece[2], piece[1])
    return piece


def _pieces() -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    by_pos: dict[tuple[int, int, int], list[int]] = {}
    for i, (pos, _) in enumerate(_FACELETS):
        by_pos.setdefault(pos, []).append(i)
    corners = [_clockwise(tuple(v)) for v in by_pos.values() if len(v) == 3]
    edges = [tuple(v) for v in by_pos.values() if len(v) == 2]
    return corners, edges


_CORNERS, _EDGES = _pieces()


def _parity(p: list[int]) -> int:
    seen, par = [False] * len(p), 0
    for i in range(len(p)):
        if seen[i]:
            continue
        j, n = i, 0
        while not seen[j]:
            seen[j] = True
            j = p[j]
            n += 1
        par ^= (n - 1) & 1
    return par


def validate(state: str) -> list[str]:
    """문제 목록을 돌려준다. 빈 목록이면 풀 수 있는 정상 상태."""
    problems: list[str] = []
    if len(state) != 54 or set(state) - set(COLORS):
        return ["54칸이 아니거나 알 수 없는 색이 있음"]
    for c in COLORS:
        if state.count(c) != 9:
            problems.append(f"{COLOR_NAMES_KO[c]} {state.count(c)}칸 (9칸이어야 함)")
    centers = [state[i * 9 + 4] for i in range(6)]
    if sorted(centers) != sorted(COLORS):
        problems.append("가운데 칸 색이 겹침")
    if problems:
        return problems

    # 가운데 칸 기준으로 색 → 면 이름. (큐브를 어느 방향으로 들고 스캔했든 통한다)
    to_face = {c: FACES[i] for i, c in enumerate(centers)}
    s = "".join(to_face[c] for c in state)

    def home(piece: tuple[int, ...]) -> tuple[str, ...]:
        return tuple(FACES[i // 9] for i in piece)

    corner_homes = [home(p) for p in _CORNERS]  # 시계 방향 순서의 면 이름
    corner_keys = [frozenset(h) for h in corner_homes]
    edge_keys = [frozenset(home(p)) for p in _EDGES]

    cperm, twist = [], 0
    for p in _CORNERS:
        cols = tuple(s[i] for i in p)
        key = frozenset(cols)
        if len(key) != 3 or key not in corner_keys:
            return ["존재할 수 없는 꼭짓점 조각"]
        j = corner_keys.index(key)
        h = corner_homes[j]
        # 색 세 개는 맞는데 도는 방향이 거울상이면 실제로는 없는 조각이다.
        if cols not in (h, h[1:] + h[:1], h[2:] + h[:2]):
            return ["존재할 수 없는 꼭짓점 조각 (거울상)"]
        cperm.append(j)
        # 꼬임: U/D 색 스티커가 이 자리의 U/D 칸에서 시계 방향으로 몇 칸 돌아갔는지.
        ref = next(k for k, i in enumerate(p) if FACES[i // 9] in "UD")
        ud = next(k for k, c in enumerate(cols) if c in "UD")
        twist += (ud - ref) % 3
    if sorted(cperm) != list(range(8)):
        return ["같은 꼭짓점 조각이 두 번 나옴"]

    eperm, flip = [], 0
    for p in _EDGES:
        cols = [s[i] for i in p]
        key = frozenset(cols)
        if len(key) != 2 or key not in edge_keys:
            return ["존재할 수 없는 모서리 조각"]
        eperm.append(edge_keys.index(key))
        flip += _edge_flipped(p, cols)
    if sorted(eperm) != list(range(12)):
        return ["같은 모서리 조각이 두 번 나옴"]

    if twist % 3:
        problems.append("꼭짓점 조각 하나가 제자리에서 돌아가 있음")
    if flip % 2:
        problems.append("모서리 조각 하나가 뒤집혀 있음")
    if _parity(cperm) != _parity(eperm):
        problems.append("조각 두 개가 서로 바뀌어 있음")
    return problems


def _edge_flipped(piece: tuple[int, ...], cols: list[str]) -> int:
    """표준 정의: U/D 색(없으면 F/B 색)이 기준 스티커 자리에 있으면 0."""
    def rank(face_name: str) -> int:
        return 0 if face_name in "UD" else 1 if face_name in "FB" else 2

    slot_faces = [FACES[i // 9] for i in piece]
    ref_slot = min(range(2), key=lambda k: rank(slot_faces[k]))
    ref_col = min(range(2), key=lambda k: rank(cols[k]))
    return int(ref_slot != ref_col)


def is_solved(state: str) -> bool:
    return all(len(set(face(state, f))) == 1 for f in FACES)
