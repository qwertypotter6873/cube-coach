"""면을 하나씩 따로 읽는 처음 상태 읽기가, 여섯 면을 정확히 읽어도 상태를 하나로 못 정하는 비율.

면끼리 어떻게 붙어 있는지는 읽지 않으므로, 여섯 면이 서로 돌린 모양인 다른 유효 상태가 있으면 가릴 수 없다
(예: R 한 번과 L' 한 번). 무작위로 k수 섞은 상태마다 여섯 면(정답)을 넣고 서로 다른 유효 상태 수를 센다.

실행: .venv/bin/python scripts/scan_ambiguity.py      출력: reports/scan_ambiguity.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from cubecoach.cube import COLORS, FACES, MOVES, SOLVED, apply, face
from cubecoach.facescan import FaceScanner

ROOT = Path(__file__).resolve().parents[1]
BASIC = [m for m in MOVES if len(m) <= 2 and m[0] in "URFDLB"]


def n_states(state: str) -> int:
    g = np.array([[COLORS.index(c) for c in face(state, s)] for s in FACES])
    return len(FaceScanner._orient(g))


def main() -> None:
    rng = random.Random(0)
    out = {"설명": "무작위 k수(같은 면 연속 없음)로 섞은 상태에서, 정확히 읽은 여섯 면이 만드는 서로 다른 유효 상태가 2개 이상인 비율",
           "결과": {}}
    for depth in [1, 2, 3, 4, 6, 8, 20]:
        n = 60 if depth < 20 else 100
        amb = 0
        for _ in range(n):
            seq: list[str] = []
            while len(seq) < depth:
                m = rng.choice(BASIC)
                if seq and seq[-1][0] == m[0]:
                    continue
                seq.append(m)
            amb += n_states(apply(SOLVED, " ".join(seq))) != 1
        out["결과"][f"{depth}수"] = {"가릴_수_없음": amb, "전체": n}
        print(depth, amb, n, flush=True)
    (ROOT / "reports" / "scan_ambiguity.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
