import random

import pytest

from cubecoach.cube import (
    COLORS,
    FACE_MOVES,
    SOLVED,
    apply,
    face,
    is_solved,
    scramble,
    validate,
)


def repeat(alg: str, n: int) -> str:
    return " ".join([alg] * n)


def test_solved_is_valid():
    assert validate(SOLVED) == []
    assert is_solved(SOLVED)


@pytest.mark.parametrize("m", ["U", "R", "F", "D", "L", "B", "x", "y", "z"])
def test_quarter_turn_has_order_four(m):
    assert apply(SOLVED, m) != SOLVED
    assert apply(SOLVED, repeat(m, 4)) == SOLVED
    assert apply(SOLVED, f"{m} {m}'") == SOLVED
    assert apply(SOLVED, f"{m}2 {m}2") == SOLVED


def test_turn_directions():
    # U를 시계 방향으로 돌리면 오른쪽 면 윗줄이 앞면으로 온다.
    assert face(apply(SOLVED, "U"), "F")[:3] == "RRR"
    # R을 돌리면 앞면 오른쪽 줄이 윗면으로 올라간다.
    assert [face(apply(SOLVED, "R"), "U")[i] for i in (2, 5, 8)] == list("GGG")
    # F를 돌리면 윗면 아랫줄이 오른쪽 면 왼쪽 줄로 간다.
    assert [face(apply(SOLVED, "F"), "R")[i] for i in (0, 3, 6)] == list("WWW")
    # D는 U와 반대: 앞면 아랫줄이 오른쪽 면으로 간다.
    assert face(apply(SOLVED, "D"), "R")[6:] == "GGG"
    # L을 돌리면 윗면 왼쪽 줄이 앞면으로 내려온다.
    assert [face(apply(SOLVED, "L"), "F")[i] for i in (0, 3, 6)] == list("WWW")
    # B를 돌리면 오른쪽 면 오른쪽 줄이 윗면 윗줄로 간다.
    assert face(apply(SOLVED, "B"), "U")[:3] == "RRR"


@pytest.mark.parametrize(
    "alg, order",
    [
        ("R U R' U'", 6),
        ("R U", 105),
        ("R U R' U R U2 R'", 6),  # Sune
        ("R U R' U' R' F R2 U' R' U' R U R' F'", 2),  # T-perm
        ("R U2 R' U' R U' R'", 6),  # Anti-Sune
    ],
)
def test_known_algorithm_orders(alg, order):
    s = SOLVED
    for i in range(1, order + 1):
        s = apply(s, alg)
        assert (s == SOLVED) == (i == order)


def test_whole_cube_rotations_keep_it_solved():
    for alg in ["x", "y", "z", "x y z'", "y2 x"]:
        s = apply(SOLVED, alg)
        assert is_solved(s) and validate(s) == []


def test_random_scrambles_stay_valid():
    rng = random.Random(0)
    for _ in range(300):
        s = apply(SOLVED, scramble(25, rng) + [rng.choice(["x", "y", "z", "x2", "y'"])])
        assert validate(s) == []
        assert all(s.count(c) == 9 for c in COLORS)


def _swap(s: str, i: int, j: int) -> str:
    lst = list(s)
    lst[i], lst[j] = lst[j], lst[i]
    return "".join(lst)


def test_impossible_states_are_rejected():
    rng = random.Random(1)
    s = apply(SOLVED, scramble(25, rng))
    # 한 칸을 잘못 읽은 경우
    wrong = "R" if s[0] != "R" else "G"
    assert validate(wrong + s[1:]) != []

    # 모서리 조각 하나 뒤집기: UF 조각의 두 칸을 맞바꾼다 (U8번 칸 = 7, F2번 칸 = 19)
    assert validate(_swap(SOLVED, 7, 19)) == ["모서리 조각 하나가 뒤집혀 있음"]
    # 꼭짓점 조각 하나 돌리기: URF 조각 세 칸을 한 칸씩 민다 (U9=8, R1=9, F3=20)
    lst = list(SOLVED)
    lst[8], lst[9], lst[20] = SOLVED[20], SOLVED[8], SOLVED[9]
    assert validate("".join(lst)) == ["꼭짓점 조각 하나가 제자리에서 돌아가 있음"]
    # 꼭짓점 두 칸만 맞바꾸면 거울상 조각이 된다
    assert validate(_swap(SOLVED, 9, 20)) == ["존재할 수 없는 꼭짓점 조각 (거울상)"]
    # 모서리 조각 두 개 맞바꾸기: UF(7,19) <-> UR(5,10)
    assert validate(_swap(_swap(SOLVED, 7, 5), 19, 10)) == ["조각 두 개가 서로 바뀌어 있음"]


def test_every_single_sticker_misread_is_caught():
    """한 칸 오인식은 100% 걸러져야 한다 — 스캔 안전망의 핵심 성질."""
    rng = random.Random(2)
    for _ in range(20):
        s = apply(SOLVED, scramble(25, rng))
        for i in range(54):
            if i % 9 == 4:
                continue
            for c in COLORS:
                if c != s[i]:
                    assert validate(s[:i] + c + s[i + 1 :]) != []


def test_face_moves_list():
    assert len(FACE_MOVES) == 18
