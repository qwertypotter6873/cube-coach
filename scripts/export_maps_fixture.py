"""TS 지도 줄이기(web/src/vision/maps.ts mapsTo)가 파이썬(cubecoach.segnet.maps_to)과 같은지 시험할 정답값.

실행: .venv/bin/python scripts/export_maps_fixture.py   → web/test/fixtures/maps_to.json
"""

import base64
import json
from pathlib import Path

import numpy as np

from cubecoach.segnet import maps_to

ROOT = Path(__file__).resolve().parents[1]


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def main() -> None:
    rng = np.random.default_rng(0)
    cases = []
    for n in (64, 48):
        logits = rng.normal(0, 2, (8, n, n))
        logp = (logits - np.log(np.exp(logits).sum(0, keepdims=True))).astype(np.float32)
        heat = (rng.uniform(0, 1, (n, n)) ** 3).astype(np.float32)
        lo, ho = maps_to(logp, heat, 48)
        cases.append({"n": n, "logp": b64(logp), "heat": b64(heat), "out_logp": b64(lo), "out_heat": b64(ho)})
    out = ROOT / "web" / "test" / "fixtures" / "maps_to.json"
    out.write_text(json.dumps({"cases": cases}))
    print(out)


if __name__ == "__main__":
    main()
