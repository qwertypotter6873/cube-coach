"""처음 상태 읽기(면 정면 스캔)의 맞추기 규칙. 브라우저판 시험은 web/test/facescan.test.ts."""

from cubecoach.cube import COLORS, FACES, SOLVED, apply, face
from cubecoach.facescan import NEED, FaceScanner


def scan(state: str) -> FaceScanner:
    sc = FaceScanner()
    for s in FACES:
        g = [COLORS.index(c) for c in face(state, s)]
        for _ in range(NEED):
            sc.add(g)
    return sc


def test_solved_and_symmetric_faces_are_read():
    # 돌려도 같은 면이 있으면 여러 방향 조합이 같은 상태를 만든다 — 같은 상태는 하나로 센다
    for alg in ["", "R2 L2 U2 D2 F2 B2", "R U R' U'"]:
        st = apply(SOLVED, alg) if alg else SOLVED
        sc = scan(st)
        assert sc.solve() == st
        assert not sc.ambiguous


def test_face_only_ambiguous_states_are_flagged():
    # R 한 번과 L' 한 번은 여섯 면이 서로 돌린 모양이라 면만으로는 가릴 수 없다
    for alg in ["R", "R U"]:
        sc = scan(apply(SOLVED, alg))
        assert sc.ready()
        assert sc.solve() is None
        assert sc.ambiguous


def test_frequent_wrong_grid_does_not_block():
    # 흰 면 로고를 파랑 가운데로 읽은 격자가 맞는 파란 면보다 많이 나와도 맞는 격자로 맞춘다
    st = apply(SOLVED, "R U F' L2 D B' R2 U' F")
    sc = scan(st)
    logo = [COLORS.index(c) for c in face(st, "U")]
    logo[4] = COLORS.index("B")
    for _ in range(10):
        sc.add(logo)
    assert sc.solve() == st
