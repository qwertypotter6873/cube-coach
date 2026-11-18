"""파이썬 추적기의 기하표와 검증용 정답값을 브라우저 추적기(TypeScript)용 JSON으로 내보낸다.

- web/src/vision/tables.json : 54칸 중심·법선, 층 소속, 회전 후보 순열, 조각 정의
- web/test/fixtures/tracker_score.json : 같은 입력에 대한 파이썬 Tracker.score 결과 (TS 이식이 같은 값을 내는지 시험)

실행: .venv/bin/python scripts/export_vision.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cubecoach.cube import _CORNERS, _EDGES, SOLVED, MOVES, to_ints, validate, apply, COLORS
from cubecoach.geometry import CENTERS, NORMALS, random_rotation
from cubecoach.tracker import AXES, CAND_EVENTS, CAND_PERMS, LAYER, Tracker

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    tables = {
        "centers": CENTERS.round(6).tolist(),
        "normals": NORMALS.round(6).tolist(),
        "axes": AXES.tolist(),
        "layer": LAYER.astype(int).tolist(),
        "candPerms": CAND_PERMS.tolist(),
        "candEvents": [list(e) for e in CAND_EVENTS],
        "corners": [list(p) for p in _CORNERS],
        "edges": [list(p) for p in _EDGES],
    }
    out = ROOT / "web" / "src" / "vision" / "tables.json"
    out.write_text(json.dumps(tables, separators=(",", ":")))
    print(out, f"{out.stat().st_size / 1024:.1f}KB")

    # 점수 함수 정답값: 무작위 지도와 무작위 가설
    rng = np.random.default_rng(0)
    n = 48
    logits = rng.normal(0, 2, (8, n, n))
    logp = logits - np.log(np.exp(logits).sum(0, keepdims=True))
    heat = rng.uniform(0, 1, (n, n)) ** 3
    colors = to_ints(apply(SOLVED, "R U F' L2 D B' R2 U' F"))
    tr = Tracker(colors)
    N = 24
    Rs = random_rotation(rng, N)
    Ts = np.stack([rng.uniform(-1, 1, N), rng.uniform(-1, 1, N), rng.uniform(9, 14, N)], axis=1)
    layers = rng.integers(-1, 6, N)
    phis = rng.uniform(-1.2, 1.2, N) * (layers >= 0)
    roi = (0.05, -0.03, 0.9)
    tr._observe(logp, heat)
    s = tr.score(Rs, Ts, layers, phis, logp, heat, roi)
    states = np.stack([colors, colors[CAND_PERMS[3]], colors[CAND_PERMS[17]]])
    s2 = tr.score(Rs, Ts, layers, phis, logp, heat, roi, colors=states)
    q_tr = Tracker(colors)
    q_tr.R, q_tr.T, q_tr.layer, q_tr.phi = Rs[0], Ts[0], -1, 0.0
    fx = {
        "logp": logp.round(6).ravel().tolist(), "heat": heat.round(6).ravel().tolist(), "n": n,
        "colors": colors.tolist(), "Rs": Rs.ravel().tolist(), "Ts": Ts.ravel().tolist(),
        "layers": layers.tolist(), "phis": phis.tolist(), "roi": list(roi), "f": tr.f,
        "peaks": {"xy": tr._peaks[0].tolist(), "col": tr._peaks[1].tolist(), "w": tr._peaks[2].tolist()},
        "score": s.tolist(), "states": states.tolist(), "score_states": s2.tolist(),
        "quality0": q_tr._quality(logp, heat, roi),
    }
    # 유효성 검사 정답값
    vrng = np.random.default_rng(1)
    cases = []
    for k in range(40):
        st = list(apply(SOLVED, " ".join(vrng.choice([m for m in MOVES if m[0] in "URFDLB"], 20))))
        if k % 4 == 1:  # 칸 하나 바꾸기
            i = int(vrng.integers(54))
            st[i] = COLORS[(COLORS.index(st[i]) + 1 + int(vrng.integers(5))) % 6]
        elif k % 4 == 2:  # 두 칸 맞바꾸기
            i, j = vrng.choice(54, 2, replace=False)
            st[i], st[j] = st[j], st[i]
        s_ = "".join(st)
        cases.append({"state": [COLORS.index(c) for c in s_], "valid": not validate(s_)})
    fx["validate"] = cases
    fpath = ROOT / "web" / "test" / "fixtures" / "tracker_score.json"
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(json.dumps(fx))
    print(fpath, f"{fpath.stat().st_size / 1024:.0f}KB", "valid cases", sum(c["valid"] for c in cases), "/", len(cases))


if __name__ == "__main__":
    main()
