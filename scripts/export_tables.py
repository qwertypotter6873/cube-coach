"""파이썬 큐브 엔진의 회전표를 웹(TypeScript) 코치 엔진용 JSON으로 내보낸다.

두 언어가 같은 표를 쓰므로, 파이썬에서 검증한 회전 정의가 브라우저에서도 그대로 성립한다.

실행: .venv/bin/python scripts/export_tables.py
결과: web/src/coach/tables.json
"""

from __future__ import annotations

import json
from pathlib import Path

from cubecoach.cube import COLORS, FACELETS, FACES, MOVES, SOLVED

OUT = Path(__file__).resolve().parents[1] / "web" / "src" / "coach" / "tables.json"


def main() -> None:
    data = {
        "faces": FACES,
        "colors": COLORS,
        "solved": [COLORS.index(c) for c in SOLVED],
        # new[i] = old[perm[i]]
        "moves": {name: [int(v) for v in perm] for name, perm in MOVES.items()},
        "facelets": [{"pos": list(p), "normal": list(n)} for p, n in FACELETS],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"{OUT} ({OUT.stat().st_size / 1024:.1f}KB), 회전 {len(MOVES)}개")


if __name__ == "__main__":
    main()
